from __future__ import print_function

import cv2
import glob
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import utils.calibrator as calibrator

import sys, os
import common

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
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
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, batch_size, fp16_on, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = batch_size
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

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    input_size = 416
    batch_size = 1
    fp16_on = False
    onnx_file_path = '../../model_data/Suspect/yolov3-tiny-suspect.onnx'
    engine_file_path = 'trt_model/yolov3-tiny-suspect_1_fp32.trt'
    
    num_classes = 3
    filters = (4 + 1 + num_classes)*3
    
    output_shapes_416 = [(batch_size, filters, 13, 13), (batch_size, filters, 26, 26)] # 2 ème variable = (5+nbr classes)*3 (255 pour coco, 33 pour key,...)
    output_shapes_480 = [(batch_size, filters, 15, 15), (batch_size, filters, 30, 30)]
    output_shapes_544 = [(batch_size, filters, 17, 17), (batch_size, filters, 34, 34)]
    output_shapes_608 = [(batch_size, filters, 19, 19), (batch_size, filters, 38, 38)]
    output_shapes_dic = {'416': output_shapes_416, '480': output_shapes_480, '544': output_shapes_544, '608': output_shapes_608}
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture("../../Datasets/test_suspect.mp4")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360) #don't work on files
    print("Width : ", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height : ", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    frame_rate_tab = []
    start_time = time.time()

    nums = 1000000

    input_resolution_yolov3_HW = (input_size, input_size)
    
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    

    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          #"yolo_masks": [(6,7,8), (3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], #tiny-yolov3-416
                          #"yolo_anchors": [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)],
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
    
    with get_engine(onnx_file_path, batch_size, fp16_on, engine_file_path) as engine, engine.create_execution_context() as context:
        # Do inference

        while(True):
            ret, frame = cap.read()

            if ret == True:
                frame_rsz = cv2.resize(frame, input_resolution_yolov3_HW, interpolation = cv2.INTER_AREA)
                frame_stream = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)
                filenames_batch.append(frame_stream)
                image_raw, image = preprocessor.process_frame(frame_stream)
                images_raw.append(image_raw)
                images.append(image)
                index += 1
                if index != nums and len(images_raw) != batch_size:
                    continue
                inputs, outputs, bindings, stream = common.allocate_buffers(engine)
                images_batch = np.concatenate(images, axis=0)
                shape_orig_WH = image_raw.size
                output_shapes = output_shapes_dic[str(input_size)]
                inputs[0].host = images_batch
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
                trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
                for i in range(len(filenames_batch)):
                    boxes, classes, scores = postprocessor.process_frame2(trt_outputs, (shape_orig_WH), i)

                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count / (end_time - start_time))
                        frame_rate_tab.append(frame_rate)
                        start_time = time.time()
                        frame_count = 0

                    frame_count += 1

                    if boxes is None:
                        det_img = frame_stream
                    else:
                        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
                        det_img = np.array(obj_detected_img)
                    cv2.putText(det_img, str(frame_rate) + " fps", (500, 50),
                        font, 1, (255, 0, 0),thickness=3, lineType=2)
                    cv2.imshow("frame", det_img)
                filenames_batch = []
                images_batch = []
                images = []
                images_raw = []
                trt_outputs = []
            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(frame_rate_tab)
        moy_FPS = np.mean(frame_rate_tab)
        print("FPS min : ", min(frame_rate_tab))
        print("FPS max : ", max(frame_rate_tab))
        print("FPS moyen :", moy_FPS)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
