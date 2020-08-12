import cv2
import time
import numpy as np
import os
import argparse

from yolo import YOLO
from PIL import Image

parser = argparse.ArgumentParser(description='Yolo predictions on images')
parser.add_argument('--model_path', 
    help='Path of Yolo Keras model (.h5)',
    default='../model_data/Suspect/yolov3-tiny-suspect.h5')
parser.add_argument('--classes_path', 
    help='Classes path',
    default='../model_data/Suspect/classes_suspect2.txt')
parser.add_argument('--img_size', 
    help='Input image size',
    default='416')
parser.add_argument('--yolo_anchors', 
    help='Tiny Yolo or Yolo anchors',
    default='../model_data/tiny_yolo_anchors.txt')
parser.add_argument('--test_dataset_path', 
    help='Images dataset path',
    default='../Datasets/Suspect/images/')
parser.add_argument('--output_pred', 
    help='Output path of Yolo predictions',
    default='../Datasets/Suspect/Predictions/Init/Yolo-Tiny/')
parser.add_argument('--output_images', 
    help='Folder\'s path with predicted bounding boxes',
    default='../Datasets/Suspect/Images_result/Init/Yolo-Tiny/')

args = parser.parse_args()

model_path = args.model_path
classes_path = args.classes_path
yolo_anchors = args.yolo_anchors
test_dataset_path = args.test_dataset_path
pred_dataset_path = args.output_pred
output_images_path = args.output_images

Yolo_Obj = YOLO(classes_path, yolo_anchors, model_path)

font = cv2.FONT_HERSHEY_SIMPLEX
moy_inf_time = 0
moy = 0

for i in os.listdir(test_dataset_path):
    
    print('Path file : ', i)
    path = i.split('.')[0]
    path = os.path.join(pred_dataset_path, path)
    annotation_path = path + '.txt'

    image_path = os.path.join(test_dataset_path,i)
    #print(image_path)
    img = cv2.imread(image_path)

    img = cv2.resize(img, (416,416), interpolation = cv2.INTER_AREA)

    if os.path.isfile(annotation_path) == True :
        os.remove(annotation_path)
        print("Delete !")

    yoloImage = Image.fromarray(img)

    start_time = time.time()

    r_image = Yolo_Obj.detect_image_ann(yoloImage, annotation_path)

    end_time = time.time()
    inference_time = int(round((end_time - start_time)*1000))
    #print('Inference time : ', inference_time)
    moy_inf_time += inference_time
    moy += 1

    result = np.asarray(r_image)

    cv2.putText(result, 'Object detection', (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object detection',result)
    path_result = os.path.join(output_images_path,str(i))
    #print(path_result)
    cv2.imwrite(path_result,result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Nombre d'images testées : ", moy)
moy_inf_time = moy_inf_time/moy
print("Moyenne temps d'inférence (par image) : ", moy_inf_time, "ms")
fps = 1/moy_inf_time*1000
print("FPS : ", fps)


