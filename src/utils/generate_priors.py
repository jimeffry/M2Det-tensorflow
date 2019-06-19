import numpy as np
from math import sqrt as sqrt
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def generate_anchors():
    image_size = cfgs.ImgSize
    anchor_scale = cfgs.AnchorScale
    anchor_configs = {}
    for idx, shape in enumerate(cfgs.FeatureShapes):
        anchor_configs[shape] = []
        for scale_octave in range(cfgs.Scales_Num):
            tmp_anchor_size = cfgs.Anchor_Size[idx][scale_octave]
            for aspect_ratio in cfgs.Ratios:
                anchor_configs[shape].append((image_size / shape,tmp_anchor_size,aspect_ratio))
                    #(image_size / shape, scale_octave / float(cfgs.Scales_Num), aspect_ratio))
    boxes_all = []
    for _, tmp_anchors in anchor_configs.items():
        boxes_level = []
        for anchor in tmp_anchors:
            stride, base_anchor_size, aspect = anchor
            #base_anchor_size = anchor_scale * stride * (2 ** octave_scale)
            anchor_size_x = base_anchor_size * sqrt(aspect)
            anchor_size_y = base_anchor_size / sqrt(aspect)
            #print('strid: {},x:{},y:{},aspect:{},size:{}'.format(stride,anchor_size_x,anchor_size_y,aspect,base_anchor_size))
            anchor_size_x_2 = anchor_size_x / 2.0
            anchor_size_y_2 = anchor_size_y / 2.0
            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        #print(np.shape(boxes_level))
        #boxes_level /= image_size
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchor_boxes = np.vstack(boxes_all)
    anchor_boxes = np.clip(anchor_boxes,0,320)
    anchor_boxes = np.array(anchor_boxes,dtype=np.float32)
    return anchor_boxes


if __name__ == '__main__':
    a = generate_anchors()
    print(a.shape)
    #print(a[10000:10005,2:]-a[10000:10005,:2])