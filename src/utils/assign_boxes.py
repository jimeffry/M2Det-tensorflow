import numpy as np
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

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

def assign_boxes(boxes, priors, num_classes, threshold=0.5):
    num_classes += 1 # add background class
    assignment = np.zeros((len(priors), 4 + num_classes + 1))
    assignment[:, 4] = 1.0 # background
    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4], priors, threshold)
    encoded_boxes = encoded_boxes.reshape(-1, len(priors), 5) # has  n boxes,so[n_box,n_priors,5]
    #print('encode:',encoded_boxes.shape)
    best_iou = encoded_boxes[:, :, -1].max(axis=0) # so get the max iou the every anchor with gd boxes
    #print('best iou',best_iou)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)# get the row mask
    #print("row,iou idx ",best_iou_idx)
    best_iou_mask = best_iou > 0 # judge by iou between prior and bbox,get the colum mask to select priors
    #print('best mask',best_iou_mask)
    best_iou_idx = best_iou_idx[best_iou_mask]
    #print("row,iou idx ",best_iou_idx)
    #so len(best_iou_idx)==len(best_iou_mask)
    assign_num = len(best_iou_idx)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    #print('mask encode box ',encoded_boxes.shape)
    # lxy so: encoded_boxes[best_iou_idx, np.arange(assign_num), :4].shape = [len(best_iou_idx),4]
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    assignment[:, 4][best_iou_mask] = 0 # background
    assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:] #the groud cls label for anchor
    assignment[:, -1][best_iou_mask] = 1 # objectness
    return assignment


if __name__=='__main__':
    gd = np.array([[100,100,200,200,0,1],[10,210,110,270,1,0]])
    bb = np.array([[105,110,205,215],[5,220,100,280],[50,50,105,105],[10,10,20,20]])
    a = assign_boxes(gd,bb,2)
    print(a)
