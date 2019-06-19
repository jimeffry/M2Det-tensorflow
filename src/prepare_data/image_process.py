# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/06/11 17:09
#project: coco detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  coco detect 
####################################################

import numpy as np
import cv2
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from generate_priors import generate_anchors

def encode_box(box, priors, assignment_threshold):
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:], box[2:])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0])
    area_gt *= (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    iou = inter / union

    encoded_box = np.zeros((len(priors), 5))
    assign_mask = iou >= assignment_threshold
    encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = priors[assign_mask] 
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    box_wh = map(float,box_wh)
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:])
    assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])
    #print(assigned_priors_wh,box_wh)
    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] /= cfgs.Variance_XY # variance0
    #print(np.maximum(box_wh / assigned_priors_wh,1e-8))
    encoded_box[:, 2:4][assign_mask] = np.log(np.maximum(box_wh / assigned_priors_wh,1e-8))
    encoded_box[:, 2:4][assign_mask] /= cfgs.Variance_WH # variance1
    return encoded_box.ravel()

def decode_boxes(boxes,priors):
    prior_width = priors[:, 2] - priors[:, 0]
    prior_height = priors[:, 3] - priors[:, 1]
    prior_center_x = 0.5 * (priors[:, 2] + priors[:, 0])
    prior_center_y = 0.5 * (priors[:, 3] + priors[:, 1])
    decode_bbox_center_x = boxes[:, 0] * cfgs.Variance_XY * prior_width # variance0
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = boxes[:, 1] * cfgs.Variance_XY * prior_height # variance0
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(boxes[:, 2] * cfgs.Variance_WH) # variance1
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(boxes[:, 3] * cfgs.Variance_WH) # variance1
    decode_bbox_height *= prior_height
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None], decode_bbox_ymin[:, None],
                                decode_bbox_xmax[:, None], decode_bbox_ymax[:, None]), axis=-1)
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0), cfgs.ImgSize)
    return decode_bbox


def assign_boxes(boxes, priors):
    threshold=cfgs.AnchorThreshold
    num_classes = cfgs.ClsNum
    num_classes += 1 # add background class
    assignment = np.zeros((len(priors), 4 + num_classes + 1))
    assignment[:, 4] = 1.0 # background
    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4], priors, threshold)
    encoded_boxes = encoded_boxes.reshape(-1, len(priors), 5) # has  n boxes,so[n_box,n_priors,5]
    #print('encode:',encoded_boxes.shape)
    best_iou = encoded_boxes[:, :, -1].max(axis=0) # so get the max iou the every anchor with gd boxes
    #print('best iou',best_iou.shape)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)# get the row mask
    #print("row,iou idx ",best_iou_idx.shape)
    best_iou_mask = best_iou > 0 # judge by iou between prior and bbox,get the colum mask to select priors
    #print('best mask',best_iou_mask.shape)
    best_iou_idx = best_iou_idx[best_iou_mask]
    #print("row,iou idx ",best_iou_idx)
    #so len(best_iou_idx)==len(best_iou_mask)
    assign_num = len(best_iou_idx)
    #print("num",best_iou_idx)
    #select the orignal iou dataes and satisfy the threshold,but not know the right cls num so get all rows
    encoded_boxes = encoded_boxes[:, best_iou_mask, :] 
    #print('mask encode box ',encoded_boxes.shape)
    #print(encoded_boxes)
    # lxy so: encoded_boxes[best_iou_idx, np.arange(assign_num), :4] is to select the anchor belong to which cls\
    # so,every assign_num has a cls num , result in best_iou_idx has assign_num length
    #print('gt_shape:',encoded_boxes[best_iou_idx, np.arange(assign_num), :4].shape)
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    assignment[:, 4][best_iou_mask] = 0 # background
    assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:] #the groud cls label for anchor
    assignment[:, -1][best_iou_mask] = 1 # objectness
    return assignment

def normalize(img):
    img = img / 255.0
    img[:,:,0] -= cfgs.PIXEL_MEAN[0]
    img[:,:,0] = img[:,:,0] / cfgs.PIXEL_NORM[0] 
    img[:,:,1] -= cfgs.PIXEL_MEAN[1]
    img[:,:,1] = img[:,:,1] / cfgs.PIXEL_NORM[1]
    img[:,:,2] -= cfgs.PIXEL_MEAN[2]
    img[:,:,2] = img[:,:,2] / cfgs.PIXEL_NORM[2]
    return img

def norm_box(img,boxes):
    img_h, img_w = img.shape[:2]
    boxes = np.array(boxes,dtype=np.float32)
    boxes[:,0] = boxes[:,0] / float(img_w)
    boxes[:,1] = boxes[:,1] / float(img_h)
    boxes[:,2] = boxes[:,2] / float(img_w)
    boxes[:,3] = boxes[:,3] / float(img_h)
    return boxes.tolist()

def scale(img, labels):
    img_size = cfgs.ImgSize
    img_h, img_w = img.shape[:2]
    ratio = max(img_h, img_w) / float(img_size)
    new_h = int(img_h / ratio)
    new_w = int(img_w / ratio)
    ox = (img_size - new_w) // 2
    oy = (img_size - new_h) // 2
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    out = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
    out[oy:oy + new_h, ox:ox + new_w, :] = scaled
    scaled_labels = []
    for label in labels:
        xmin, ymin, xmax, ymax = label[0:4]
        xmin = (xmin * new_w + ox) #/ img_size
        ymin = (ymin * new_h + oy) #/ img_size
        xmax = (xmax * new_w + ox) #/ img_size
        ymax = (ymax * new_h + oy) #/ img_size
        label = [xmin, ymin, xmax, ymax] + label[4:]
        scaled_labels.append(label)
    return out, np.asarray(scaled_labels,dtype=np.int32)

def augment(img, boxes):
    boxes = norm_box(img,boxes)
    img, boxes = scale(img, boxes)
    img = normalize(img)
    return img, boxes

def label_show(img,boxes):
    for box in boxes:
        box = map(int,box)
        cv2.rectangle(img,pt1=(box[0],box[1]),pt2=(box[2], box[3]),color=(0,0,255))

def process_imgs(imgs,gt_labels,num_obj,batch_size,anchors):
    imgs = np.array(imgs)
    gt_labels = np.array(gt_labels,dtype=np.int32)
    num_obj = np.array(num_obj,dtype=np.int32)
    imgs_out = []
    gt_label_box = []
    #print('img',num_obj.shape,num_obj[0])
    for idx in range(batch_size):
        tmp_img = imgs[idx]
        tmp_obj = num_obj[idx]
        tmp_labels = gt_labels[idx][:tmp_obj,:]
        bbox = tmp_labels[:,:4]
        label = tmp_labels[:,4]
        onehot = np.zeros([tmp_obj,cfgs.ClsNum])
        positive = np.ones(tmp_obj)
        onehot[:,label] = positive
        box_label = np.concatenate([bbox,onehot],axis=1)
        img, boxes = augment(tmp_img, box_label)
        assignment = assign_boxes(boxes, anchors)
        imgs_out.append(img)
        gt_label_box.append(assignment)
    return np.array(imgs_out,dtype=np.float32),np.array(gt_label_box,dtype=np.float32)

if __name__=='__main__':
    img = cv2.imread("../../data/innsbruck.png")
    bb = [[100,100,200,200,1],[210,10,310,50,2]]
    print(np.shape(bb))
    img_c = img.copy()
    #label_show(img_c,bb)
    img_bat = np.expand_dims(img,0)
    bb_bat = [bb]
    priors = generate_anchors()
    img_out,bb_out = process_imgs(img_bat,bb_bat,2,1,priors)
    #print(bb_out)
    print(img_out[0,0,:10,0])
    print(img_out[0,-5:-1,-10,1])
    img_out = np.array(img_out[0],dtype=np.uint8)
    #label_show(imga,box)
    cv2.imshow('src',img_c)
    cv2.imshow('aug',img_out)
    cv2.waitKey(0)