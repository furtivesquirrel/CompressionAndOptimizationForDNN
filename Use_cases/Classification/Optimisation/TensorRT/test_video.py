import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import os
import time
import cv2
import argparse

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input


parser = argparse.ArgumentParser(
    prog='Classification - TRT model evaluation on video',
    description='This script evaluates performances of model on video')

parser.add_argument('--trt_model_path', 
    help='Optimized TRT model\'s path (.pb)',
    default='trt_model/trt_graph_FP32.pb')

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

model_path = args.trt_model_path 
classes_path = args.classes_path
source_vid =  args.source_video  # source_vid = 0 pour ouvrir webcam
img_size = int(args.img_size)


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph(model_path)

output_names = ['dense_2/Softmax']
input_names = ['input_1']

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
#print("image_size: {}".format(image_size))

input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

#print("input_tensor_name: {}\noutput_tensor_name: {}".format(
#    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


#test_dataset_path ="fire_dataset/"

classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(source_vid)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) #Just for webcam
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360) #Just for webcam

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

        feed_dict = {
            input_tensor_name: x
        }

        preds = tf_sess.run(output_tensor, feed_dict)
        preds.shape

        pred  = np.squeeze(preds)
        #print(pred)

        end_time = time.time()

        if (end_time - start_time) > fps_display_interval:
            frame_rate = int(frame_count / (end_time - start_time))
            inf_time = round(time.time() - start_time2)
            frame_rate_tab.append(frame_rate)
            inf_times.append(inf_time)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame_stream, str(frame_rate) + " fps", (500, 50),
            font, 1, (255, 0, 0),thickness=3, lineType=2)

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
print(inf_times)
moy_FPS = np.mean(frame_rate_tab)
print("FPS min : ", min(frame_rate_tab))
print("FPS max : ", max(frame_rate_tab))
print("FPS moyen :", moy_FPS)
cap.release()
cv2.destroyAllWindows()
tf_sess.close()


