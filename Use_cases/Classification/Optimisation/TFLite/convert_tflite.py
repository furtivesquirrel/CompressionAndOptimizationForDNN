import tensorflow as tf
import numpy as np
import cv2
import argparse
import glob

parser = argparse.ArgumentParser(
    prog='Classification - Keras-TFlite conversion',
    description='TFlite conversion of Keras model')

parser.add_argument('--model_path', 
    help='Keras model path (.h5)',
    default='../../model_data/MobileNet_final.h5')

parser.add_argument('--output_path', 
    help='Output model path (.tflite)',
    default='model-default.tflite')

args = parser.parse_args()

### BASIC CONVERSION 
converter = tf.lite.TFLiteConverter.from_keras_model_file(args.model_path) 

### FP16 : DEFAULT 
'''converter = tf.lite.TFLiteConverter.from_keras_model_file(args.model_path) 
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]'''

### FP16 : OPTIMIZE FOR SIZE

'''converter = tf.lite.TFLiteConverter.from_keras_model_file(args.model_path) 
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]'''

tfmodel = converter.convert() 
open(args.output_path , "wb").write(tfmodel)
