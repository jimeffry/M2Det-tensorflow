import os
import sys
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__), '../network'))
from m2det import M2Det
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from generate_priors import generate_anchors
from nms import soft_nms, nms
from classes import get_classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../prepare_data'))
from image_process import normalize,decode_boxes

class Detector:
    def __init__(self, model_path, input_size, use_sfam, threshold):
        self.model_path = model_path
        self.input_size = input_size
        self.use_sfam = use_sfam
        self.threshold = threshold
        self.priors = generate_anchors()
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        self.net = M2Det(self.inputs, tf.constant(False), self.use_sfam)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        ratio = max(img_h, img_w) / float(self.input_size)
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.input_size - new_w) // 2
        oy = (self.input_size - new_h) // 2
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        inp = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8) * 127
        inp[oy:oy + new_h, ox:ox + new_w, :] = scaled
        #inp = (inp - 127.5) / 128.0
        inp = normalize(inp)
        return inp, ox, oy, new_w, new_h

    def detect(self, img):
        img_h, img_w = img.shape[:2]
        inp, ox, oy, new_w, new_h = self.preprocess(img)
        inp = np.expand_dims(inp,0)
        outs = self.sess.run(self.net.prediction, feed_dict={self.inputs: inp})
        outs = outs[0]
        #print('out,',outs.shape)
        boxes = decode_boxes(outs[:, :4],self.priors)
        preds = np.argmax(outs[:, 4:], axis=1)
        confidences = np.max(outs[:, 4:], axis=1)
        #print('preds',preds.shape,confidences[:5])
        #print(preds[:5])
        # skip background class
        mask = np.where(preds > 0)
        #print('cls_mask',mask[0].shape)
        boxes = boxes[mask]
        preds = preds[mask]
        confidences = confidences[mask]

        mask = np.where(confidences >= cfgs.AnchorThreshold)
        #print('confidence mask and threshold',mask[0].shape,cfgs.AnchorThreshold)
        boxes = boxes[mask]
        preds = preds[mask]
        confidences = confidences[mask]
        #print('final score:',confidences)
        results = []
        for box, clsid, conf in zip(boxes, preds, confidences):
            xmin, ymin, xmax, ymax = box
            left = int((xmin - ox) / new_w * img_w)
            top = int((ymin - oy) / new_h * img_h)
            right = int((xmax - ox) / new_w * img_w)
            bottom = int((ymax - oy) / new_h * img_h)
            conf = float(conf)
            name, color = get_classes(clsid - 1)
            results.append({
                'left': left,
                'top': top,
                'right': right, 
                'bottom': bottom,
                'name': name,
                'color': color,
                'confidence': conf,
            })
        #results = nms(results,1-self.threshold,'Min')
        results = nms(results,self.threshold)
        results = nms(results,1- self.threshold,'Min')
        #results = soft_nms(results, self.threshold)
        #print("after nms result num: ",len(results))
        return results
