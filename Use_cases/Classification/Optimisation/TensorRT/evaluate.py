import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import os
import time
import argparse

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input


parser = argparse.ArgumentParser(
    prog='Classification - TRT model evaluation on BDD',
    description='This script evaluates performances of model on BDD')

parser.add_argument('--trt_model_path', 
    help='Optimized TRT model\'s path (.pb)',
    default='trt_model/trt_graph_FP32.pb')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='../../model_data/classes_fire.txt')

parser.add_argument('--test_dataset_path', 
    help='Test dataset\' path (folder)',
    default='../../Dataset/')

args = parser.parse_args()

trt_model_path = args.trt_model_path
test_dataset_path = args.test_dataset_path
classes_path = args.classes_path



def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph(trt_model_path)

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
print("image_size: {}".format(image_size))

input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

inf_times = []
fps = []
moy = 0
test_precision = 0
start_time_fps = time.time()

for i in os.listdir(test_dataset_path):
    dir_path = os.path.join(test_dataset_path,i)
    #print(i)

    for j in os.listdir(dir_path):
        #print(j)

        image_path = os.path.join(dir_path,j)
        #print(image_path)

        img = image.load_img(image_path, target_size=image_size[:2])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feed_dict = {
            input_tensor_name: x
        }

        start_time = time.time()

        preds = tf_sess.run(output_tensor, feed_dict)
        preds.shape

        pred  = np.squeeze(preds)
        #print(pred)

        end_time = time.time()

        inference_time = int(round((end_time - start_time)*1000))
        #print('Inference time : ',inference_time)
        inf_times.append(inference_time)

        #print(moy_inf_time)
        moy += 1
        #print(moy)

        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])

        if result[0][0] == i:
            #print("OK")
            test_precision += 1


        #img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #plt.imshow(img)
        #plt.show()

print("Nombre d'images testées : ", moy)
print("Images bien classées : ", test_precision)

test_precision = (test_precision/moy)*100
print("Précision de test : ", test_precision, "%")

#moy_inf_time = moy_inf_time/moy
moy_inf_time = np.array(inf_times).mean()
print("Moyenne temps d'inférence (par image) : ", moy_inf_time, "ms")

fps = 1/moy_inf_time*1000
print("FPS : ", fps)

tf_sess.close()


