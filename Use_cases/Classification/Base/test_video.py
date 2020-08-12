import cv2
import time
import numpy as np
import argparse

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

parser = argparse.ArgumentParser(
    prog='Classification - Evaluation on video',
    description='This script evaluates performances of model on video')

parser.add_argument('--model_path', 
    help='Model\'s path (.h5)',
    default='../model_data/MobileNet_final.h5')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='../model_data/classes_fire.txt')

parser.add_argument('--video_source', 
    help='Video source (0) for webcam',
    default="../Video/test_vid2.mp4")

parser.add_argument('--img_size', 
    help='Model\'s image input size',
    default='224')


args = parser.parse_args()

img_size = int(args.img_size)
model = load_model(args.model_path)
video_source = args.video_source
classes_path = args.classes_path

classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(video_source)
print("Width : ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height : ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0
start_time = time.time()
start_time2 = time.time()
frame_rate_tab = []
inf_times = []

while(True):
    ret, frame = cap.read()

    if ret == True:
        frame_rsz = cv2.resize(frame, (img_size,img_size), interpolation = cv2.INTER_AREA)
        frame_stream = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)

        x = image.img_to_array(frame_rsz)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = model.predict(x)[0]

        end_time = time.time()

        if (end_time - start_time) > fps_display_interval:

            frame_rate = int(frame_count / (end_time - start_time))
            frame_rate_tab.append(frame_rate)
            inf_time = round(time.time() - start_time2)
            inf_times.append(inf_time)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame_stream, str(frame_rate) + " fps", (500, 50),
            font, 1, (255,0,0),thickness=3, lineType=2)

        frame_count += 1

        

        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])

        if result[0][0]=='fire':
            cv2.putText(frame_stream, 'Fire', (10,320), font, 1, (51,0,204), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame_stream,'No fire', (10,320), font, 1, (51,255,51), 2, cv2.LINE_AA)
    else:
        break

    cv2.putText(frame_stream, 'Fire detection', (10,50), font, 1.3, (255,0,0), 3, cv2.LINE_AA)
    cv2.imshow('Fire detection', frame_stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(frame_rate_tab)
#print(inf_times)
moy_FPS = np.mean(frame_rate_tab)
print("FPS min : ", min(frame_rate_tab))
print("FPS max : ", max(frame_rate_tab))
print("FPS moyen :", moy_FPS)
cap.release()
cv2.destroyAllWindows()
