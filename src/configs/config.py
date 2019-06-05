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
cfgs.DataSet_Name = 'COCO'
# Training config**********************************************
cfgs.ClsNum = 80
cfgs.Variance_XY = 0.1
cfgs.Variance_WH = 0.2
cfgs.AnchorBoxes = 19215 # (40x40+20x20+10x10+5x5+3x3+1x1)x9=19215
# anchors setting***********************************************
cfgs.Scales_Num = 3 #2**, 1, 2, 4
cfgs.Ratios = [(1, 1), (1.41, 0.71), (0.71, 1.41)]
cfgs.FeatureShapes=[40, 20, 10, 5, 3, 1]
cfgs.ImgSize = 320
cfgs.AnchorScale = 2.0