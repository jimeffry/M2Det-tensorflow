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
from data import Data
from loss import calc_loss
from generate_priors import generate_anchors
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from read_rfrecord import Read_Tfrecord
from image_process import process_imgs

def parms():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
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
    parser.add_argument('--sfam', action='store_true', default=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tfrecord_dir',type=str,default='../../data')
    return parser.parse_args()

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
    #********************************************creat logging
    logger = logging.getLogger()
    hdlr = logging.FileHandler(os.path.join(log_dir,time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
    #formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(threadName)-10s] %(message)s')
    #hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    #********************************************load data
    with tf.name_scope("Load_data"):
        '''
        databox = Data(image_dir=args.image_dir, 
                    label_dir=args.label_dir, 
                    assignment_threshold=args.assignment_threshold)
        databox.start()
        dataset_size = databox.size
        logger.info('Dataset size: {}'.format(dataset_size))
        '''
        '''
        y_true_size = 4 + num_classes + 1 + 1 = num_classes + 6
        * 4 => bbox coordinates (x1, y1, x2, y2);
        * num_classes + 1 => including a background class;
        * 1 => denotes if the prior box was matched to some gt boxes or not;
        '''
        tfrd = Read_Tfrecord(cfgs.DataSet_Name,data_record_dir,batch_size,1000)
        num_obj_batch, img_batch, gtboxes_label_batch = tfrd.next_batch()
        anchors = tf.py_func(generate_anchors,[],tf.float32)
        anchors.set_shape([None,4])
    #********************************************build network
    y_true_size = cfgs.ClsNum + 6
    inputs = tf.placeholder(tf.float32, [None, cfgs.ImgSize, cfgs.ImgSize, 3])
    y_true = tf.placeholder(tf.float32, [None, cfgs.AnchorBoxes, y_true_size])
    is_training = tf.constant(True)
    net = M2Det(inputs, is_training, sfam_fg)
    y_pred = net.prediction
    with tf.name_scope("Losses"):
        total_loss = calc_loss(y_true, y_pred, box_loss_weight=box_loss_scale)
        tf.summary.scalar('cls_box/cb_loss', total_loss)
        weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'kernel' in v.name]
        decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights])) * 1e-3
        tf.summary.scalar('weight/weight_loss', decay)
        total_loss += decay
        tf.summary.scalar('total/total_loss', total_loss)
    #***************************************************************build trainer
    with tf.name_scope("optimizer"):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.piecewise_constant(global_step,
                                        boundaries=[np.int64(x) for x in cfgs.DECAY_STEP],
                                        values=[y for y in cfgs.LR])
        tf.summary.scalar('lr', lr)
        #optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        train_var = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            #opt = tf.train.AdamOptimizer(learning_rate=lr)
            grads = tf.gradients(total_loss, train_var)
            #tf.summary.histogram("Grads/")
            train_op = opt.apply_gradients(zip(grads, train_var), global_step=global_step)
    #***********************************************************************************training
    with tf.name_scope("training_op"):
        sess = tf.Session()
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
            assert tf.train.get_checkpoint_state(model_dir),'the params dictionary is not valid'
            model_path = "%s-%s" %(model_path,load_num)
            saver.restore(sess, model_path)
            logger.info('Resuming training %s' % model_path)
        # build summary
        summary_op = tf.summary.merge_all()
        summary_path = os.path.join(log_dir,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
        # begin to tain
        try:
            for epoch_tmp in range(epoches):
                for step in range(np.ceil(cfgs.Train_Num/batch_size).astype(np.int32)):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    #x_batch, t_batch = databox.get(batch_size)
                    x_batch,t_batch,obj_batch = sess.run([img_batch,gtboxes_label_batch,num_obj_batch])
                    x_batch,t_batch = tf.py_func(process_imgs,[x_batch,t_batch,obj_batch,batch_size,anchors],[tf.float32,tf.float32])
                    global_value = sess.run(global_step)
                    if step % cfgs.SHOW_TRAIN_INFO !=0 and step % cfgs.SMRY_ITER !=0:
                        _ = sess.run([train_op], feed_dict={inputs: x_batch, y_true: t_batch})
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO ==0:
                            _, loss_value,cur_lr = sess.run([train_op, total_loss,opt._lr], feed_dict={inputs: x_batch, y_true: t_batch})
                            logger.info('{} \t epoch:{}, lr:{}, step: {}, loss: {}'.format(str(training_time),epoch_tmp,cur_lr,global_value, loss_value))
                        if step % cfgs.SMRY_ITER ==0:
                            _, summary_str = sess.run([train_op,summary_op], feed_dict={inputs: x_batch, y_true: t_batch})
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
