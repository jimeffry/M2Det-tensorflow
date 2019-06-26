import numpy as np


def nms(results,threshold,mode='Union'):
    outputs = []
    rectangles = []
    if len(results) >0:
        for result in results:
            rectangles.append([result['left'], result['top'], result['right'], result['bottom'],result['confidence']])
        boxes = np.array(rectangles)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s  = boxes[:,4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort())
        pick = []
        #I[-1] have hightest prob score, I[0:-1]->others
        #i = 0
        while len(I)>0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) 
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if mode == 'Min':
                ot = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                ot = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(ot<=threshold)[0]]
            #if i==10:
             #   print('nms:',ot)
            #i+=1
        #result_rectangle = boxes[pick].tolist()
        outputs = [results[idx] for idx in pick]
    return outputs

def calc_iou(box1, box2):
    # box: left, top, right, bottom
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if w <= 0 or h <= 0:
        return 0
    intersection = w * h
    s1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = max(s1 + s2 - intersection, 1e-5)
    iou = intersection / union
    return iou

def nms_(results, iou_threshold=0.4):
    '''
    Args:
        results: [num_boxes, Dict]
            Dict: left, top, right, bottom, name, color, confidence
        outputs: [K, Dict]
            K: the number of valid boxes
            Dict: left, top, right, bottom, name, color, confidence
    '''

    outputs = []
    while len(results) > 0:
        ix = np.argmax([result['confidence'] for result in results])
        result = results[ix]
        outputs.append(result)
        del results[ix]

        box1 = [result['left'], result['top'], result['right'], result['bottom']]
        to_delete = []
        for jx in range(len(results)):
            box2 = [results[jx]['left'], results[jx]['top'], results[jx]['right'], results[jx]['bottom']]
            iou = calc_iou(box1, box2)
            if iou >= iou_threshold:
                to_delete.append(jx)
        for jx in to_delete[::-1]:
            del results[jx]

    return outputs

def soft_nms(results, threshold, sigma=0.5):
    '''
    Args:
        results: [num_boxes, Dict]
            Dict: left, top, right, bottom, name, color, confidence
        outputs: [K, Dict]
            K: the number of valid boxes
            Dict: left, top, right, bottom, name, color, confidence
    '''
    outputs = []
    while len(results) > 0:
        ix = np.argmax([result['confidence'] for result in results])
        result = results[ix]
        outputs.append(result)
        del results[ix]

        # soft-nms with a Gaussian penalty function
        box1 = [result['left'], result['top'], result['right'], result['bottom']]
        to_delete = []
        for jx in range(len(results)):
            box2 = [results[jx]['left'], results[jx]['top'], results[jx]['right'], results[jx]['bottom']]
            iou = calc_iou(box1, box2)
            penalty = np.e ** (-iou ** 2 / sigma)
            results[jx]['confidence'] = results[jx]['confidence'] * penalty
            if results[jx]['confidence'] < threshold:
                to_delete.append(jx)

        for jx in to_delete[::-1]: # delete from biger to small
            del results[jx]

    return outputs
