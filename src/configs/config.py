# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/06/5 10:09
#project: 
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  
####################################################
from easydict import EasyDict

cfgs = EasyDict()
# dataset *************************************************
cfgs.BIN_DATA = 0
cfgs.DATASET_LIST = ['COCO','VOC']
cfgs.DataSet_Name = 'COCO'
cfgs.ClsNum = 8
cfgs.DataNames = ['person','bicycle','car','motorcycle','bus','truck','traffic_light','stop_sign']
cfgs.PIXEL_MEAN = [0.485,0.456,0.406] # R, G, B
cfgs.PIXEL_NORM = [0.229,0.224,0.225] #rgb
# Training config**********************************************
cfgs.Train_Num = 97481 #117266
cfgs.Variance_XY = 0.1
cfgs.Variance_WH = 0.2
cfgs.AnchorBoxes = 12810 # (40x40+20x20+10x10+5x5+3x3+1x1)x6=12810
cfgs.SHOW_TRAIN_INFO = 1000
cfgs.SMRY_ITER = 2000
cfgs.ModelPrefix = 'model'
cfgs.LR = [0.01,0.001,0.0005,0.0001,0.00001]
cfgs.DECAY_STEP = [35000, 70000,105000,140000]
# anchors setting***********************************************
cfgs.Scales_Num = 2 #2**, 1, 2, 4
cfgs.Ratios = [1.0/2.0,1.0, 2.0] #[(1, 1), (1.41, 0.71), (0.71, 1.41)]
cfgs.FeatureShapes=[40, 20, 10, 5, 3, 1]
#cfgs.Anchor_Minsize = [25.6, 48.0, 105.6, 163.2, 220.8, 278.4]
#cfgs.Anchor_Maxsize =  [48.0, 105.6, 163.2, 220.8, 278.4, 336.0]
cfgs.Anchor_Size = [(25.6,48.0),(48.0,105.6),(105.6,163.2),(163.2,220.8),(220.8,278.4),(278.4,336.0)]
cfgs.ImgSize = 320
cfgs.IMG_SIZE = [320,320]
cfgs.AnchorScale = 2.0
cfgs.AnchorThreshold = 0.5