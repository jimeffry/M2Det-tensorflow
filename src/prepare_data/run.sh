#!/bin/bash
#python convert_data_to_tfrecord.py --VOC-dir /home/lxy/Downloads/DataSet/VOC_dataset/VOCdevkit/VOC2012  --xml-dir Annotations --image-dir JPEGImages \
 #           --save-name train --dataset-name VOC2012  
#widerface
python convert_data_to_tfrecord.py    --image-dir /data/COCO/train2017 \
            --save-name train --dataset-name COCO --anno-file ../../data/CoCo/coco2017.txt --save-dir /data/train_record