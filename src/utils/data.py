import cv2
import glob
import os
import multiprocessing
import time
import numpy as np
import sys
from generate_priors import generate_priors
from assign_boxes import assign_boxes
from augment import augment
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

class Data:
    def __init__(self, image_dir, label_dir, assignment_threshold):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_classes = cfgs.ClsNum 
        self.input_size = cfgs.ImgSize
        self.assignment_threshold = assignment_threshold
        self.priors = generate_priors()
        self.size = self.get_size()

    def start(self):
        self.q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.put, args=(self.q,))
        p.start()

    def get_paths(self):
        paths = []
        for bb_path in glob.glob(os.path.join(self.label_dir, '*.txt')):
            im_path = os.path.join(self.image_dir, os.path.splitext(os.path.basename(bb_path))[0] + '.jpg')
            if os.path.exists(im_path):
                paths.append([im_path, bb_path])
        return paths

    def get_size(self):
        return len(self.get_paths())

    def put(self, q):
        queue_max_size = 1000
        paths = []
        while True:
            if len(paths) == 0:
                paths = self.get_paths()
            if q.qsize() >= queue_max_size:
                time.sleep(0.1)
                continue

            ix = np.random.randint(0, len(paths))
            path = paths.pop(ix)
            im_path, bb_path = path
            npimg = np.fromfile(im_path, dtype=np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(bb_path) as f:
                lines = f.read().splitlines()

            boxes = []
            for line in lines:
                ix, xmin, ymin, xmax, ymax = line.split('\t') 
                onehot_label = np.eye(self.num_classes)[int(ix)]
                box = [float(xmin), float(ymin), float(xmax), float(ymax)] + onehot_label.tolist()
                boxes.append(box)

            img, boxes = augment(img, boxes, self.input_size)

            if len(boxes) == 0:
                continue
            boxes = np.array(boxes)
            assignment = assign_boxes(boxes, self.priors, self.num_classes, self.assignment_threshold)
            q.put([img, assignment])
            
    def get(self, batch_size):
        x_batch = []
        t_batch = []
        for _ in range(batch_size):
            while True:
                if self.q.qsize() == 0:
                    time.sleep(1)
                    continue
                img, assignment = self.q.get()
                x_batch.append(img)
                t_batch.append(assignment)
                break
        return np.asarray(x_batch), np.asarray(t_batch)
