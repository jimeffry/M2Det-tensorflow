import numpy as np
import tensorflow as tf
import logging
import time
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from m2det import M2Det
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
#from data import Data
from loss import calc_loss
from generate_priors import generate_anchors
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from read_tfrecord import Read_Tfrecord
from image_process import process_imgs

def parms():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='../../models/')
    parser.add_argument('--log_dir', default='../../logs/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scale', type=float, default=20.0)
    parser.add_argument('--load_num', type=str, default=None)
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--assignment_threshold', type=float, default=0.5)
    parser.add_argument('--save_weight_period', type=int, default=5) # (40x40+20x20+10x10+5x5+3x3+1x1)x9=19215
    parser.add_argument('--shapes', type=int, nargs='+', default=[40, 20, 10, 5, 3, 1]) # for 320x320
    parser.add_argument('--sfam', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tfrecord_dir',type=str,default='../../data')
    parser.add_argument('--tower_name',type=str,default='tower')
    return parser.parse_args()


def average_gradients(tower_gradients):
    average_grads = []
    # Run this on cpu_device to conserve GPU memory
    with tf.device('/cpu:0'):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []
            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])
            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)
    # Return result to caller
    return average_grads

def main(args):
    #******************************************** load args
    batch_size = args.batch_size
    data_record_dir = args.tfrecord_dir
    log_dir = args.log_dir
    sfam_fg = args.sfam
    box_loss_scale = args.scale
    model_dir = args.model_dir
    load_num = args.load_num
    epoches = args.epoches
    save_weight_period = args.save_weight_period
    gpu_list = [int(i) for i in args.gpu.split(',')]
    gpu_num = len(gpu_list)
    #********************************************creat logging
    logger = logging.getLogger()
    hdlr = logging.FileHandler(os.path.join(log_dir,time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    #formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(threadName)-10s] %(message)s')
    #hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.info("train gpu:{}".format(gpu_list))
    #********************************************load data
    y_true_size = cfgs.ClsNum + 6
    with tf.name_scope("Load_data"):
        tfrd = Read_Tfrecord(cfgs.DataSet_Name,data_record_dir,batch_size,1000)
        num_obj_batch, img_batch, gtboxes_label_batch = tfrd.next_batch()
        anchors = tf.py_func(generate_anchors,[],tf.float32)
        anchors.set_shape([None,4])
        x_batch,y_true = tf.py_func(process_imgs,[img_batch,gtboxes_label_batch,num_obj_batch,batch_size,anchors],[tf.float32,tf.float32])
        x_batch.set_shape([None,cfgs.ImgSize,cfgs.ImgSize,3])
        y_true.set_shape([None,cfgs.AnchorBoxes,y_true_size])
        images_s = tf.split(x_batch, num_or_size_splits=gpu_num, axis=0)
        labels_s = tf.split(y_true, num_or_size_splits=gpu_num, axis=0)
    #***************************************************************build trainer
    with tf.name_scope("optimizer"):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.piecewise_constant(global_step,
                                        boundaries=[np.int64(x) for x in cfgs.DECAY_STEP],
                                        values=[y for y in cfgs.LR])
        #tf.summary.scalar('lr', lr)
        #opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
    #*****************************************************************get multi-model net
    # Calculate the gradients for each model tower.
    tower_grads = []
    loss_scalar_dict = {}
    loss_hist_dict = {}
    all_ave_cls_loss = tf.Variable(0.0, name='all_ave_cls_loss', trainable=False)
    all_ave_bbox_loss = tf.Variable(0.0,name='all_ave_bbox_loss',trainable=False)
    is_training = tf.constant(True)
    for i,idx in enumerate(gpu_list):
        with tf.variable_scope(tf.get_variable_scope(),reuse= i > 0):
            with tf.device('/gpu:%d' % idx):
                with tf.name_scope('%s_%d' % (args.tower_name, idx)) as scope:
                    net = M2Det(images_s[i], is_training, sfam_fg)
                    y_pred = net.prediction
                    #resue
                    tf.get_variable_scope().reuse_variables()
                    total_loss,bbox_loss,class_loss = calc_loss(labels_s[i], y_pred, box_loss_weight=box_loss_scale)
                    loss_scalar_dict['cls_box/cb_loss_%d' % idx] = total_loss
                    ave_box_loss = tf.reduce_mean(bbox_loss)
                    all_ave_bbox_loss.assign_add(ave_box_loss)
                    ave_clss_loss = tf.reduce_mean(class_loss)
                    all_ave_cls_loss.assign_add(ave_clss_loss)
                    weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'kernel' in v.name]
                    decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights])) * 5e-5
                    loss_scalar_dict['weight/weight_loss_%d' % idx] = decay
                    total_loss.assign_add(decay)
                    loss_scalar_dict['total/total_loss_%d' % idx] = total_loss
                    loss_scalar_dict['bbox/ave_box_loss_%d' % idx] = ave_box_loss
                    loss_scalar_dict['class/ave_class_loss_%d' % idx] = ave_clss_loss
                    loss_hist_dict['bbox/box_loss_%d' % idx] = bbox_loss
                    loss_hist_dict['class/class_loss_%d' % idx] = class_loss
                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
                    #tf.add_to_collection("total_loss",total_loss)
    #************************************************************************************compute gradients
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)
        #train_loss = tf.reduce_mean(tf.get_collection("total_loss"),0)
        #train_op = optimizer.minimize(train_loss,colocate_gradients_with_ops=True)
    #*****************************************************************************************add summary
    summaries = []
    # add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # add loss summary
    for keys, val in loss_scalar_dict.items():
        summaries.append(tf.summary.scalar(keys, val))
    for keys, val in loss_hist_dict.items():
        summaries.append(tf.summary.histogram(keys, val))
    # add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)
    #***********************************************************************************training
    with tf.name_scope("training_op"):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        sess = tf.Session(config=tf_config)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=10)
        #load model
        model_path = os.path.join(model_dir,cfgs.DataSet_Name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,cfgs.ModelPrefix)
        if load_num is not None :
            #assert tf.train.get_checkpoint_state(model_dir),'the params dictionary is not valid'
            model_path = "%s-%s" %(model_path,load_num)
            saver.restore(sess, model_path)
            logger.info('Resuming training %s' % model_path)
        # build summary
        #summary_op = tf.summary.merge_all()
        summary_path = os.path.join(log_dir,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
        # begin to tain
        try:
            for epoch_tmp in range(epoches):
                for step in range(np.ceil(cfgs.Train_Num/batch_size).astype(np.int32)):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    global_value = sess.run(global_step)
                    if step % cfgs.SHOW_TRAIN_INFO !=0 and step % cfgs.SMRY_ITER !=0:
                        _ = sess.run([train_op])
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO ==0:
                            _, loss_value,cur_lr,ave_box,ave_cls = sess.run([train_op, total_loss,opt._lr,all_ave_bbox_loss,all_ave_cls_loss])
                            logger.info('{} \t epoch:{}, lr:{}, step: {}, loss: {} , bbox:{}, cls:{}'.format(str(training_time),epoch_tmp,cur_lr,global_value, loss_value,ave_box/gpu_num,ave_cls/gpu_num))
                        if step % cfgs.SMRY_ITER ==0:
                            _, summary_str = sess.run([train_op,summary_op])
                            summary_writer.add_summary(summary_str,global_value)
                            summary_writer.flush()
                if (epoch_tmp > 0 and epoch_tmp % save_weight_period == 0) or (epoch_tmp == epoches - 1):
                    dst = model_path
                    saver.save(sess, dst, epoch_tmp,write_meta_graph=False)
                    logger.info(">>*************** save weight ***: %d" % epoch_tmp)
        except tf.errors.OutOfRangeError:
            print("Trianing is error")
        finally:
            coord.request_stop()
            summary_writer.close()
            coord.join(threads)
            #record_file_out.close()
            sess.close()

if __name__ == '__main__':
    args = parms()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)