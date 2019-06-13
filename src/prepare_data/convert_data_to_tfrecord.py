# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import argparse
import os 
import sys
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
import transform


def parms():
    parser = argparse.ArgumentParser(description='dataset convert')
    parser.add_argument('--VOC-dir',dest='VOC_dir',type=str,default='../../data/',\
                        help='dataset root')
    parser.add_argument('--xml-dir',dest='xml_dir',type=str,default='VOC_XML',\
                        help='xml files dir')
    parser.add_argument('--image-dir',dest='image_dir',type=str,default='VOC_JPG',\
                        help='images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='../../data/',\
                        help='tfrecord save dir')
    parser.add_argument('--save-name',dest='save_name',type=str,\
                        default='train',help='image for train or test')
    parser.add_argument('--img-format',dest='img_format',type=str,\
                        default='.jpg',help='image format')
    parser.add_argument('--dataset-name',dest='dataset_name',type=str,default='VOC',\
                        help='datasetname')
    #for widerface
    parser.add_argument('--anno-file',dest='anno_file',type=str,\
                        default='../../data/wider_gt.txt',help='annotation files')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='datasetname')
    return parser.parse_args()

class DataToRecord(object):
    def __init__(self,save_path):
        self.writer = tf.python_io.TFRecordWriter(path=save_path)

    def _int64_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self,value):
        """Wrapper for insert float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def write_recore(self,img_dict):
        # maybe do not need encode() in linux
        img_name = img_dict['img_name']
        img_height,img_width = img_dict['img_shape']
        img = img_dict['img_data']
        '''
        img[:,:,0] = img[:,:,0] - cfgs.PIXEL_MEAN[0] # R
        img[:,:,1] = img[:,:,1] - cfgs.PIXEL_MEAN[1] # G
        img[:,:,2] = img[:,:,2] - cfgs.PIXEL_MEAN[2] # B
        img = img / cfgs.PIXEL_NORM
        '''
        if not cfgs.BIN_DATA:
            img_raw = cv2.imencode('.jpg', img)[1]
            img = img_raw
        gtbox_label = img_dict['gt']
        #num_objects = img_dict['num_objects']
        feature = tf.train.Features(feature={
            'img_name': self._bytes_feature(img_name),
            'img_height': self._int64_feature(img_height),
            'img_width': self._int64_feature(img_width),
            'img': self._bytes_feature(img.tostring()),
            'gtboxes_and_label': self._bytes_feature(gtbox_label.tostring()), #self._int64_feature(gtbox_label),
            'num_objects': self._int64_feature(gtbox_label.shape[0])
        })
        example = tf.train.Example(features=feature)
        self.writer.write(example.SerializeToString())
    def close(self):
        self.writer.close()


class Data2TFrecord(object):
    def __init__(self,args):
        self.anno_file = args.anno_file
        save_dir = args.save_dir
        dataset_name = cfgs.DataSet_Name #args.dataset_name
        self.image_dir = args.image_dir
        save_name = args.save_name
        self.img_format = args.img_format
        save_path = os.path.join(save_dir,dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = save_name + '.tfrecord'
        record_save_path = os.path.join(save_path,save_name)
        self.record_w = DataToRecord(record_save_path)
        self.property_file = os.path.join(save_path,'property.txt')
        self.cls_num = len(cfgs.DataNames)
    #convert widerface data to tfrecord
    def rd_anotation(self,annotation):
        '''
        annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
        '''
        img_dict = dict()
        annotation = annotation.strip().split(',')
        self.img_prefix = annotation[0]
        #boxed change to float type
        bbox = map(float, annotation[1:])
        gt_box_labels = np.array(bbox,dtype=np.int32).reshape(-1,5)
        self.boxes = gt_box_labels[:,:4]
        self.labels = gt_box_labels[:,4]
        #onehot_label = np.eye(self.num_classes)[int(ix)]
        #gt
        #self.boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        #label = np.ones([self.boxes.shape[0],1],dtype=np.int32)*NAME_LABEL_MAP['face']
        #gt_box_labels = np.concatenate((self.boxes,label),axis=1)
        #load image
        img_path = os.path.join(self.image_dir, self.img_prefix + self.img_format)
        if not os.path.exists(img_path):
            return None
        self.img_org = cv2.imread(img_path)
        if self.img_org is None:
            return None
        img_shape = self.img_org.shape[:2]
        #img = img[:,:,::-1]
        if cfgs.BIN_DATA:
            img_raw = open(img_path,'rb').read()
        num_objects_one_img = gt_box_labels.shape[0]
        #gt_box_labels = np.reshape(gt_box_labels,-1)
        #gt_list = gt_box_labels.tolist()
        self.img_name = img_path.split('/')[-1]
        img_dict['img_data'] = img_raw if cfgs.BIN_DATA else self.img_org
        img_dict['img_shape'] = img_shape
        img_dict['gt'] = gt_box_labels #gt_list
        img_dict['img_name'] = self.img_name
        img_dict['num_objects'] = num_objects_one_img
        return img_dict

    def transform_imgbox(self):
        '''
        annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
        '''
        auger_list=["Sequential", "Fliplr","Affine","Dropout", \
                    "AdditiveGaussianNoise","SigmoidContrast","Multiply"]
        trans = transform.Transform(img_auger_list=auger_list)
        img_dict = dict()
        if self.img_org is None:
            print("aug img is None")
            return None
        img_aug,boxes_aug,keep_idx = trans.aug_img_boxes(self.img_org,[self.boxes.tolist()])
        if not len(boxes_aug) >0:
            #print("aug box is None")
            return None
        img_data = img_aug[0]
        boxes_trans = np.array(boxes_aug[0], dtype=np.int32).reshape(-1, 4)
        label = np.array(self.labels[keep_idx[1][0]],dtype=np.int32).reshape(-1,1)
        #label = np.ones([boxes_trans.shape[0],1],dtype=np.int32)*NAME_LABEL_MAP['face']
        #print('box',boxes_trans.shape)
        #print('label',np.shape(label))
        gt_box_labels = np.concatenate((boxes_trans,label),axis=1)
        num_objects_one_img = gt_box_labels.shape[0]
        #gt_box_labels = np.reshape(gt_box_labels,-1)
        #gt_list = gt_box_labels.tolist()
        img_dict['img_data'] = img_data
        img_dict['img_shape'] = img_data.shape[:2]
        img_dict['gt'] = gt_box_labels #gt_list
        img_dict['img_name'] = self.img_prefix+'_aug'+self.img_format
        img_dict['num_objects'] = num_objects_one_img
        return img_dict

    def convert_img_to_tfrecord(self):
        '''
        anno_file = kargs.get('anno_file',None)
        save_dir = kargs.get('save_dir',None)
        dataset_name = kargs.get('dataset_name',None)
        image_dir = kargs.get('image_dir',None)
        save_name = kargs.get('save_name',None)
        img_format = kargs.get('img_format',None)
        #property_file = kargs.get('property_file',None)
        '''
        failed_aug_path = open('aug_failed.txt','w')
        property_w = open(self.property_file,'w')
        anno_p = open(self.anno_file,'r')
        anno_lines = anno_p.readlines()
        total_img = 0
        dataset_img_num = len(anno_lines)
        cnt_failed = 0
        for count,tmp in enumerate(anno_lines):
            img_dict = self.rd_anotation(tmp)
            if img_dict is None:
                print("the img path is none:",tmp.strip().split()[0])
                continue
            self.record_w.write_recore(img_dict)
            #label_show(img_dict,'bgr')
            total_img+=1
            if random.randint(0, 1) and not cfgs.BIN_DATA:
                img_dict = self.transform_imgbox()
                if img_dict is None:
                    #print("the aug img path is none:",tmp.strip().split()[0])
                    failed_aug_path.write(tmp.strip().split()[0] +'\n')
                    cnt_failed+=1
                    continue
                self.record_w.write_recore(img_dict)
                #label_show(img_dict,'bgr')
                total_img+=1
            view_bar('Conversion progress', count + 1,dataset_img_num)
        print('\nConversion is complete!')
        print('total img:',total_img)
        print("aug failed:",cnt_failed)
        property_w.write("{},{}".format(len(cfgs.DataNames),total_img))
        property_w.close()
        self.record_w.close()
        anno_p.close()
        failed_aug_path.close()

def label_show(img_dict,mode='rgb'):
    img = img_dict['img_data']
    if mode == 'rgb':
        img = img[:,:,::-1]
    img = np.array(img,dtype=np.uint8)
    gt = img_dict['gt']
    num_obj = img_dict['num_obj']
    #print("img",img.shape)
    #print("box",gt.shape)
    for i in range(num_obj):
        #for rectangle in gt:
        rectangle = gt[i]
        #print('show bbl',rectangle)
        #print(map(int,rectangle[5:]))
        score_label = str("{}".format(rectangle[4]))
        #score_label = str(1.0)
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        if len(rectangle) > 5:
            for i in range(5,15,2):
                cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("img",img)
    cv2.waitKey(0)

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


if __name__ == '__main__':
    args = parms()
    dataset = args.dataset_name
    ct = Data2TFrecord(args)
    ct.convert_img_to_tfrecord()
        
