import time

import cv2

import numpy as np
import torch


def extract_bounding_box(frame, path):
    """Extracts bounding box using model in path"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path)  # default
    detect_obj = model(frame[..., ::-1])
    results = detect_obj.pandas().xyxy[0].to_dict(orient='records')
    if results:

        results__ = np.array(
            [[int(results_['xmin']), int(results_['ymin']), int(results_['xmax']), int(results_['ymax'])] for results_
             in results]).flatten()
    else:
        results__ = np.array([0, 0, 0, 0])
    rec_coords = np.array(results__)
    x1, y1, x2, y2 = rec_coords[0], rec_coords[1], rec_coords[2], rec_coords[3]
    return x1, y1, x2, y2
    # x = (x1 + x2) / 2  # x center
    # y = (y1 + y2) / 2  # y center
    # w = x2 - x1  # width
    # h = y2 - y1  # height
    # return np.array([x, y, w, h])
    #
    # return np.array([x, y, w, h])
    #      results]).flatten()


import torch.backends.cudnn as cudnn

cam = cv2.VideoCapture(0)
path = r'C:\Users\Research\Obj_Rec\yolov5\best.pt'
cudnn.benchmark = True
while True:
    _, image = cam.read()
    # time.sleep(2.0)
    l, t, r, b = extract_bounding_box(image, path)
    cv2.rectangle(image, (l, t), (r, b), (255, 0, 255), 2)
    cv2.imshow('out', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
