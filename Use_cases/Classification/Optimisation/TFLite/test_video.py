import cv2
import time
import numpy as np
import tensorflow as tf
import argparse

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input


parser = argparse.ArgumentParser(
    prog='Classification - TRT model evaluation on video',
    description='This script evaluates performances of model on video')

parser.add_argument('--tflile_model_path', 
    help='Compressed TFlite model\'s path (.tflite)',
    default='model-default-FP16.tflite')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='../../model_data/classes_fire.txt')

parser.add_argument('--source_video', 
    help='Video source',
    default='../../Video/test_vid2.mp4')

parser.add_argument('--img_size', 
    help='Model\'s image input size',
    default='224')

args = parser.parse_args()

model_path = args.tflile_model_path
classes_path = args.classes_path
source_vid =  args.source_video  # source_vid = 0 pour ouvrir webcam
img_size = int(args.img_size)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter = tf.contrib.lite.Interpreter(model_path="exposure.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(source_vid)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0
frame_rate_tab = []
start_time = time.time()

while(True):
    ret, frame = cap.read()

    if ret == True:
        frame_rsz = cv2.resize(frame, (img_size,img_size), interpolation = cv2.INTER_AREA)
        frame_stream = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)

        x = image.img_to_array(frame_rsz)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        input_data = np.array(x, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data)

        pred  = np.squeeze(output_data)
        #print(pred)
        
        end_time = time.time()
        #print(end_time-start_time)

        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            frame_rate_tab.append(frame_rate)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame_stream, str(frame_rate) + " fps", (500, 50),
            font, 1, (255,0,0), thickness=3, lineType=2)

        frame_count += 1

        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])

        if result[0][0]=='fire':
            cv2.putText(frame_stream, 'Fire', (10,330), font, 1, (51,0,204), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame_stream,'No fire', (10,330), font, 1, (51,255,51), 2, cv2.LINE_AA)
    else:
        break

    cv2.putText(frame_stream, 'Fire detection', (10, 50), font, 1.3, (255,0,0), 3, cv2.LINE_AA)
    cv2.imshow('Fire detection',frame_stream)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(frame_rate_tab)
moy_FPS = np.mean(frame_rate_tab)
print("FPS min : ", min(frame_rate_tab))
print("FPS max : ", max(frame_rate_tab))
print("FPS moyen :", moy_FPS)
cap.release()
cv2.destroyAllWindows()