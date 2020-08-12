# CompressionAndOptimizationForDNN
DNN compression methods for Nvidia Jetson  : TensorFlow-TensorRT and TFLite for classification &amp; TensorRT for localization


## Create environment
```
pip install virtualenv
virtualenv name_env --python=3.6
source ~/name_env/bin/activate
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev libblas3 liblapack3 liblapack-dev libblas-dev gfortran libfreetype6-dev
pip install -r requirements.txt
sudo pip3 install −−extra-index-url https ://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION 'tensorflow<2'
```
##### Link TensorRT : 
```
cd /usr/src/tensorrt/samples/python
export PYTHONPATH=/usr/lib/python3.6/dist-packages :$PYTHONPATH
```
##### Link OpenCV : 
```
cd env/lib/python3.6/site-packages
ln -s /usr/lib/python3.6/dist-packages/cv2/python-3.6/ cv2.cpython-36m-aarch64-linuxgnu.so cv2.cpython-36m-aarch64-linux-gnu.so
```


## Methods

### Classification
`cd Use_cases/Classification/`

#### Native
1. Download Keras model and put it in `model_data/`
2. Create a Test set and put it in `Datasets/`
3. Model evaluation (in `Base/`)
      - Test on one image : Run `python test_img.py` with command line option `--model_path`, `−−classes_path`, `−−img_size`, `−−img_path` and `−−num_classes`
      - Test on dataset : Run `python evaluate.py` with command line option `--model_path`, `−−classes_path`, `−−img_size` and `−−test_dataset_path `
      - Test on video : Run `python test_video.py` with command line option `--model_path`, `−−classes_path`, `−−img_size` and `−−video_source `

#### TensorRT
In `Optimisation/TensorRT` : 
1. Freeze graph : run `python freeze_graph.py` with command line option `--model_path` and `−−output_path`
  If execution problem, run notebook in Colab and put TensorFlow graph (.pb) in `Optimisation/TensorRT/tf_model`
2. Optimize TensorFlow graph with TensorRT
      - In `trt_optimisation.py`, change `max_batch_size`, `max_workspace_size_bytes` and `precision_mode` (FP32, FP16, INT8) in line 31 to 33
      - Run `python trt_optimisation.py` with command line option `--pb_model_path` and `−−trt_output_path`
3. Model evaluation
      - Test on dataset : Run `python evaluate.py` with command line option `-−trt_model_path`, `−−classes_path` and `−−test_dataset_path`
      - Test on video : Run `python  test_video.py` with command line option `-−trt_model_path`, `−−classes_path`, `−−source_video` and `−−img_size`  

#### TFLite
In `Optimisation/TFLite` :
1. Modify `convert_tflite.py` to select your optimization
2. Run `python convert_tflite.py` with command line option `−−model_path` and `−−output_path`
3. Model evaluation
      - Test on dataset : Run `python evaluate.py` with command line option `-−tflite_model_path`, `−−classes_path`, `−−test_dataset_path` and  `−−img_size`
      - Test on video : Run `python  test_video.py` with command line option `-−tflite_model_path`, `−−classes_path`, `−−source_video` and `−−img_size`
  
### Localization
`cd Use_cases/Localisation/`

#### Native 
1. Download YOLOv3 or YOLOv3-tiny DarkNet weights and cfg on https://pjreddie.com/darknet/yolo/ or use yours and put in `model_data/`
2. Retrained if you want using Keras framework with TensorFLow backend and put in `model_data/`
3. You can run `python `
4. Put .txt classes file in `model_data`
5. Create a test set with its ground-truth and put it in `Datasets/images/` and `Datasets/ground-truth/`. You can use annotations tools to help you (in Datasets/Tools/) :
      - `split_annotations.py` to split an entire file annotation
      - `convert_annotations.py` to convert annotations in <x><y><width><height> format to <left><top><right><bottom>
      - `reshape_img.py` to change size of all images in a folder
6. For rest of codes, you need files .weights, .cfg and .h5. If you don't have all you can use script conversion in `Localisation/Conversion/`
      - `python darknet_to_Keras`
      - `python Keras_to_Darknet`
7. Evaluate predictions on images :
      - run `python Base/predict_images.py` with command line option `-−model_path`, `−−classes_path`, `−−img_size`, `--yolo_anchors`, `−−test_dataset_path`, `−−output_pred`, `−−−output_images`
      - run `python mAP/map.py` with command line option `-−gtfolder`, `−−detfolder` and `−−savepath`
8. Evaluate on video : run `python Base/test_video.py` with command line option `-−model_path`, `−−classes_path`, `--yolo_anchors` and `−−video_source`
  
  #### TensorRT
  1. Convert DarkNet weights to ONNX (It could be do on PC): 
     - In `Conversion/YoloV3_weights_to_onnx/`, run `pip install -r requirements.txt` to get onnx requirements
     - Run `python yolov3_to_onnx.py` with command line option `-−model_path`, `−−classes_path`, `−−config_path`, `--img_size` and `−−output_onnx_model`
2. Go in folder `Optimisation/TensorRT/`
3. In `data_processing.py`, modify classes_path at line 11 and the number of classes at line 16
4. Create TRT engine : 
      - Choose precision mode (FP32, FP16 or INT8)
              - For FP32 and FP16, put lines 65 and 66 in comment and uncomment 64. For FP32 change `fp16_on` in line 143 to False and to True for FP16 
              - For INT8, comment line 64 and uncomment 65 and 66. 

      - Put ma_batch_size at line 144
      - Lines 129 to 135, comment lines you don't need relative to the model you choose
      - Run `python onnx_to_tensorrt.py` with command line option `-−input_size`, `−−onnx_file_path`, `−−engine_file_path`, `--num_classes`, `−−dataset_path`, pred_dataset_path` and `−−result_images_path`
5. Evaluate with mAP

