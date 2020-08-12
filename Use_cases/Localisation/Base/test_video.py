# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import argparse

from yolo import YOLO
from PIL import Image

parser = argparse.ArgumentParser(
    prog='Localisation - Evaluation on video',
    description='This script evaluates performances of model on video')

parser.add_argument('--model_path', 
    help='Model\'s path (.h5)',
    default='../model_data/Suspect/yolov3-tiny-suspect.h5')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='../model_data/Suspect/classes_suspect.txt')

parser.add_argument('--yolo_anchors', 
    help='Classes path (.txt)',
    default='../model_data/tiny_yolo_anchors.txt')

parser.add_argument('--video_source', 
    help='Video source (0) for webcam',
    default="../Datasets/test_suspect.mp4")



args = parser.parse_args()

model_path = args.model_path
classes_path = args.classes_path
yolo_anchors = args.yolo_anchors
video_source = args.video_source


Yolo_Obj = YOLO(classes_path, yolo_anchors, model_path)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(video_source)
print("Width : ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height : ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0
start_time = time.time()
frame_rate_tab = []

while(True):
    ret, frame = cap.read()
    
    if ret == True:
        frame_rsz = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)
        yoloImage = Image.fromarray(frame_rsz)
        r_frame = Yolo_Obj.detect_image(yoloImage)

        end_time = time.time()

        if (end_time - start_time) > fps_display_interval:

            frame_rate = int(frame_count / (end_time - start_time))
            frame_rate_tab.append(frame_rate)
            start_time = time.time()
            frame_count = 0

        frame_count += 1

        result = np.asarray(r_frame)

        cv2.putText(result, str(frame_rate) + " fps", (500, 50),
            font, 1, (0, 255, 0),thickness=2, lineType=2)

        cv2.putText(result, 'Suspect object detection', (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    else:
        break
    
    cv2.imshow('Suspect object detection', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(frame_rate_tab)
moy_FPS = np.mean(frame_rate_tab)
print("FPS min : ", min(frame_rate_tab))
print("FPS max : ", max(frame_rate_tab))
print("FPS moyen :", moy_FPS)

cap.release()
cv2.destroyAllWindows()
