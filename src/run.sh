#!/usr/bin/bash
#python mscoco/process.py --image_dir /data/COCO/train2017 --annotation_path /data/COCO/annotations/instances_train2017.json --output_dir ../data/ --out_file coco2017.txt

############################## run train
#python train/train.py --image_dir /wdc/LXY.data/CoCo2017/train2017 --label_dir ../data/COCO  --model_dir /wdc/LXY.data/models/tf_models/ --log_dir ../logs/ \
 #   --epoches 30 --save_weight_period 5 --gpu 1

 ########################run demotest2017/000000000619.jpg
python test/demo.py --inputs /data/COCO/train2017/000000550601.jpg  --model_dir /data/models/m2det   --load_num 6 --threshold 0.001 