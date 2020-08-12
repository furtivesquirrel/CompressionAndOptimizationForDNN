from matplotlib import pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input #input shape=299x299
from keras.applications.mobilenet import preprocess_input 

import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser(
    prog='Classification - Evaluation on video',
    description='This script evaluates performances of model on video')

parser.add_argument('--model_path', 
    help='Model\'s path (.h5)',
    default='model_data/MobileNet_final.h5')

parser.add_argument('--classes_path', 
    help='Classes path (.txt)',
    default='model_data/classes_fire.txt')

parser.add_argument('--image_path', 
    help='Image\'s path',
    default="Base/feu.jpg")

parser.add_argument('--img_size', 
    help='Model\'s image input size',
    default='224')

parser.add_argument('--num_classes', 
    help='Number of classes',
    default='2')

args = parser.parse_args()

classes_path = args.classes_path
print("NUM : ", len(classes_path))
image_path=args.image_path 
img_size = int(args.img_size)
num_classes = int(args.num_classes)
model = load_model(args.model_path)

top_n=num_classes # nombre de classes

# load class names
classes = []
with open(classes_path, 'r') as f:
    classes = list(map(lambda x: x.strip(), f.readlines()))

img = image.load_img(image_path, target_size=(img_size,img_size)) #attention, changer
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict
pred = model.predict(x)[0]
result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
result.sort(reverse=True, key=lambda x: x[1])

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
font = cv2.FONT_HERSHEY_COMPLEX 

for i in range(top_n):
    (class_name, prob) = result[i]
    print(result[i])
    textsize = cv2.getTextSize(class_name, font, 1, 2)[0]
    textX = (img.shape[1] - textsize[0]) / 2
    textY = (img.shape[0] + textsize[1]) / 2
    if (i == 0) :
        cv2.putText(img, class_name, (int(textX)-100, int(textY)), font, 10, (255,255,255), 6, cv2.LINE_AA)
    print("Top %d ====================" % (i + 1))
    print("Class name: %s" % (class_name))
    print("Probability: %.2f%%" % (prob))
plt.imshow(img)
plt.show()