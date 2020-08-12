from __future__ import print_function

import glob
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import utils.calibrator as calibrator

from PIL import ImageDraw
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
import common

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, annotation_path, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    #print(bboxes, confidences, categories)
    #print("ICI", zip(bboxes, confidences, categories))
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))
        predicted_class = all_categories[category]

        with open(annotation_path, "a") as f:
                ann = """{pred_class} {score} {left} {top} {right} {bottom}""".format(
                    pred_class=predicted_class, score=score, left=left, top=top,
                    right=right, bottom=bottom)
                f.write(ann + "\n")

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, max_batch_size, fp16_on, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_on
            #builder.int8_mode = True
            #builder.int8_calibrator = calibrator.Yolov3EntropyCalibrator(data_dir="../../Datasets/Suspect/images/", cache_file='INT8CacheFile')
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def download_file(path, link, checksum_reference=None):
    if not os.path.exists(path):
        print('downloading')
        wget.download(link, path)
        print()
    if checksum_reference is not None:
        raise ValueError('error')
    return path
def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    parser = argparse.ArgumentParser(
        prog='ONNX to TensorRT conversion',
        description='Convert the Yolo ONNX model to TensorRT')

    parser.add_argument('--input_size', 
        help='Input size model',
        default='416')

    parser.add_argument('--onnx_file_path', 
        help='ONNX model\'s path (.onnx)',
        default='../../model_data/Suspect/yolov3-suspect.onnx')

    parser.add_argument('--engine_file_path', 
        help='TensorRT engine\'s path (.trt)',
        default='trt_model/yolov3-suspect_2_fp32.trt')

    parser.add_argument('--num_classes', 
        help='Number of classes',
        default='3')

    parser.add_argument('--dataset_path', 
        help='Path of the folder Dataset',
        default='../../Datasets/Suspect/images-416/')

    parser.add_argument('--pred_dataset_path', 
        help='Output path of Yolo predictions',
        default='../../Datasets/Suspect/Predictions/TensorRT/Yolo-Tiny-128/')

    parser.add_argument('--result_images_path', 
        help='Path of images with predict bounding box',
        default='../../Datasets/Suspect/Images_result/TensorRT/Yolo-Tiny-128/')

    args = parser.parse_args()

    input_size = int(args.input_size)
    onnx_file_path = args.onnx_file_path
    engine_file_path = args.engine_file_path
    num_classes = int(args.num_classes)
    test_dataset_path = args.dataset_path
    save_path = args.result_images_path
    pred_dataset_path = args.pred_dataset_path

    fp16_on = False
    batch_size = 2

    filters = (4 + 1 + num_classes)*3
    
    output_shapes_416 = [(batch_size, filters, 13, 13), (batch_size, filters, 26, 26)] # 2 ème variable = (5+nbr classes)*3 (255 pour coco, 33 pour key,...)
    output_shapes_480 = [(batch_size, filters, 15, 15), (batch_size, filters, 30, 30)]
    output_shapes_544 = [(batch_size, filters, 17, 17), (batch_size, filters, 34, 34)]
    output_shapes_608 = [(batch_size, filters, 19, 19), (batch_size, filters, 38, 38)]
    output_shapes_dic = {'416': output_shapes_416, '480': output_shapes_480, '544': output_shapes_544, '608': output_shapes_608}

    filenames = glob.glob(os.path.join(test_dataset_path, '*.jpg'))
    
    nums = len(filenames)

    input_resolution_yolov3_HW = (input_size, input_size)
    
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    
    output_shapes = output_shapes_dic[str(input_size)]

    postprocessor_args = {#"yolo_masks": [(3, 4, 5), (0, 1, 2)], #tiny
                          "yolo_masks": [(6,7,8), (3, 4, 5), (0, 1, 2)], 
                          #"yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], #tiny-yolov3
                          "yolo_anchors": [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], #YoloV3
                          "obj_threshold": 0.5, 
                          "nms_threshold": 0.35,
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    
    # Do inference with TensorRT
    filenames_batch = []
    images = []
    images_raw = []
    trt_outputs = []
    index = 0
    moy_inf_time = 0
    moy = 0

    with get_engine(onnx_file_path, batch_size, fp16_on, engine_file_path) as engine, engine.create_execution_context() as context:
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        for filename in filenames:
            #print("Path file : ", filename)
            #path = filename.split('.')[4]
            #path2 = path.split('/')[4]
            #print("PATH: ", path2)
            #name_ann = os.path.join(pred_dataset_path, path2)
            #annotation_path = name_ann + '.txt'
            #print("ANNOTATION : ", annotation_path)
            filenames_batch.append(filename)

            '''if os.path.isfile(annotation_path) == True :
                os.remove(annotation_path)
                print("Delete !")'''

            image_raw, image = preprocessor.process(filename)
            images_raw.append(image_raw)
            images.append(image)
            index += 1

            if index != nums and len(images_raw) != batch_size:
                continue

            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            images_batch = np.concatenate(images, axis=0)
            inputs[0].host = images_batch   
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
            t2 = time.time()
            t_inf = int(round((t2 - t1)*1000))
            #print("Inf time : ",t_inf)
            moy_inf_time += t_inf
            #print("MOY : ", moy)

            print(len(trt_outputs))
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
            for i in range(len(filenames_batch)):
                fname = filenames_batch[i].split('/')
                fname = fname[-1].split('.')[0]
                print(fname)
                name_ann = os.path.join(pred_dataset_path, fname)
                annotation_path = name_ann + '.txt'
                #print("ANNOTATION : ", annotation_path)
                if os.path.isfile(annotation_path) == True :
                    os.remove(annotation_path)
                    print("Delete !")
                img_raw = images_raw[i]
                #print(img_raw)
                shape_orig_WH = img_raw.size
                print("SHAPE : ", shape_orig_WH)
                
                boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH), i)
                
                if boxes is not None:
                    print("boxes size:", len(boxes))
                else:
                    continue
		        # Draw the bounding boxes onto the original input image and save it as a PNG file
                obj_detected_img = draw_bboxes(img_raw, boxes, scores, classes, ALL_CATEGORIES, annotation_path)
                output_image_path = save_path + fname + '_' + str(input_size) + '_bboxes.png'
                obj_detected_img.save(output_image_path, 'PNG')
                print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
            
            filenames_batch = []
            images_batch = []
            images = []
            images_raw = []
            trt_outputs = []
    print(len(filenames))
    moy_inf_time = moy_inf_time/len(filenames)
    print("Moyenne temps d'inférence (par image) : ", moy_inf_time, "ms")
    fps = 1/moy_inf_time*1000
    print("FPS : ", fps)

if __name__ == '__main__':
    main()
