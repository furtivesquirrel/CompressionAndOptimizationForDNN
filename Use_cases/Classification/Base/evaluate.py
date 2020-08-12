import cv2
import time
import numpy as np
import os
import matplotlib
import argparse

from matplotlib import pyplot as plt
from PIL import Image

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

parser = argparse.ArgumentParser(
    prog='Classification - Evaluation on BDD',
    description='This script evaluates performances of model on BDD')

parser.add_argument('--model_path', 
    help='Model\'s path (.h5)',
    default='model_data/MobileNet_final.h5')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='model_data/classes_fire.txt')

parser.add_argument('--img_size', 
    help='Model\'s image input size',
    default='224')

parser.add_argument('--test_dataset_path', 
    help='Test dataset\' path (folder)',
    default='Dataset/')

args = parser.parse_args()

img_size = int(args.img_size)
model = load_model(args.model_path)

classes = []
with open(args.classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

font = cv2.FONT_HERSHEY_SIMPLEX
inf_times = []
moy = 0
test_precision = 0
test_data = []

for i in os.listdir(args.test_dataset_path):
    dir_path = os.path.join(args.test_dataset_path,i)
    #print(i)
    if i == 'fire':
        label = np.array([1,0])
    else:
        label = np.array([0,1])

    for j in os.listdir(dir_path):
        #print(j)
        image_path = os.path.join(dir_path,j)
        img = image.load_img(image_path, target_size=(img_size,img_size)) #change with required size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        test_data.append([np.array(x), label])

        start_time = time.time()

        pred = model.predict(x)[0]
        #print(pred)

        end_time = time.time()
        inference_time = (end_time - start_time)*1000
        #print('Inference time : ',inference_time)
        inf_times.append(inference_time)


        moy += 1

        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])
        #print(result)

        if result[0][0] == i:
            #print("OK")
            test_precision += 1

        #img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #plt.imshow(img)
        #plt.show()

testImages = np.array([i[0] for i in test_data]).reshape(-1, img_size, img_size, 3)
testLabels = np.array([i[1] for i in test_data])

print("Nombre d'images testées : ", moy)
print("Images bien classées : ", test_precision)

loss, acc = model.evaluate(testImages, testLabels, verbose=0)
print("Acc (précision de test): ", acc * 100)
print("Loss", loss)

#test_precision = (test_precision/moy)*100
#print("Précision de test : ", test_precision, "%")

#moy_inf_time = moy_inf_time/moy
moy_inf_time = np.array(inf_times).mean()
print("Moyenne temps d'inférence (par image) : ", moy_inf_time, "ms")

fps = 1/moy_inf_time*1000
print("FPS : ", fps)





