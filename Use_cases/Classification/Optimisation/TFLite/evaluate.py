import tensorflow as tf
import numpy as np
import cv2
import os 
import time
import argparse

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image


parser = argparse.ArgumentParser(
    prog='Classification - TFlite model evaluation on BDD',
    description='This script evaluates performances of model on BDD')

parser.add_argument('--tflite_model_path', 
    help='Compressed TFlite model\'s path (.tflite)',
    default='model-default-FP16.tflite')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='../../model_data/classes_fire.txt')

parser.add_argument('--test_dataset_path', 
    help='Test dataset\' path (folder)',
    default='../../Dataset/')

parser.add_argument('--img_size', 
    help='Model\'s image input size',
    default='224')

args = parser.parse_args()

tflite_model_path = args.tflite_model_path
test_dataset_path = args.test_dataset_path
classes_path = args.classes_path
img_size = int(args.img_size)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

inf_times = []
moy = 0
test_precision = 0

for i in os.listdir(test_dataset_path):
    dir_path = os.path.join(test_dataset_path,i)
    #print(i)

    for j in os.listdir(dir_path):
        #print(j)

        image_path = os.path.join(dir_path,j)
        #print(image_path)

        img = image.load_img(image_path, target_size=(img_size,img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        input_data = np.array(x, dtype=np.float32)

        start_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data)

        pred  = np.squeeze(output_data)
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
        print(result)

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
