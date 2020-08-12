import configparser
import io
import argparse
import numpy as np

from yolo import YOLO
from collections import defaultdict
from keras.models import load_model


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream

def main():

    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # major, minor, revision=[0,2,0] seen=32013312
    m_revision=[0,2,0]
    seen=[32013312]
    # convert to  bytes
    m_revision_const = np.array(m_revision,dtype=np.int32)
    m_revision_bytes=m_revision_const.tobytes()

    seen_const=np.array(seen,dtype=np.int64)
    seen_bytes=seen_const.tobytes()

    print('write revision information\n')
    weight_file.write(m_revision_bytes)
    weight_file.write(seen_bytes)

    # conv2d and batch_normalize layers
    b=0
    print('Start write weights\n')
    for section in cfg_parser.sections():

        #print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            # get 'convolutional_'
            num = int(section.split('_')[-1])+1
            # get 'batch_normalize'
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            # if batch_normalize write it three times and  activation='leaky'
            if batch_normalize:
                # from batch_normalization layer extract bn_weight_list
                batch_weight_name = 'batch_normalization_' + str(num-b)
                bn_weight_list_layer=model.get_layer(batch_weight_name)
                bn_weight_list =bn_weight_list_layer.get_weights()

                # from bn_weight_list extract bn_weight and con_bias
                conv_bias = bn_weight_list[1]
                bn_weight = [bn_weight_list[0], bn_weight_list[2], bn_weight_list[3]]

                # from conv2d layer extract conv_weight
                conv2d_weight_name = 'conv2d_' + str(num)
                # print conv2d_weight_name
                print(conv2d_weight_name,'\n')
                print(batch_weight_name, '\n')
                conv2d_weight_name_layer=model.get_layer(conv2d_weight_name)
                # list[ndarray]
                conv_weight = conv2d_weight_name_layer.get_weights()
                conv_weight=conv_weight[0]
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight=bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape,'\n')

                # convert bn_weight to bytes then write to file
                bn_weight_array=np.array(bn_weight,dtype=np.float32)
                bytes_bn_weight=bn_weight_array.tobytes()
                weight_file.write(bytes_bn_weight)
                print(bn_weight_array.shape,'\n')

                conv_weight_array=np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight=conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)
                print(conv_weight_array.shape,'\n')

            # not  existence batch_normalize layers, write it two times
            else:
                # b is disorder parameter
                b+=1
                # from conv2d layer extract conv_weight（include conv_bias)
                print('\n')
                conv2d_weight_name = 'conv2d_' + str(num)
                print('disorder',conv2d_weight_name,'\n\n')
                conv2d_weight_name_layer = model.get_layer(conv2d_weight_name)
                conv_weights =conv2d_weight_name_layer.get_weights()

                # extract conv_bias conv2d_weight
                conv_bias = conv_weights[-1]
                conv_weight = conv_weights[0]
                conv_weight=np.array(conv_weight)
                # transpose
                conv_weight = np.transpose(conv_weight, [3, 2, 0, 1])

                # write the file with order conv_bias、conv2d_weight
                # conv_bias convert to  bytes
                bias_weight = np.array(conv_bias,dtype=np.float32)
                bytes_bias_weight = bias_weight.tobytes()
                weight_file.write(bytes_bias_weight)
                print(bias_weight.shape)
                # conv_weight convert to bytes
                conv_weight_array = np.array(conv_weight,dtype=np.float32)
                bytes_conv_weight = conv_weight_array.tobytes()
                weight_file.write(bytes_conv_weight)
                # pritn the shape
                print(conv_weight_array.shape)

    weight_file.close()
    print("Convert success !\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Convert Keras yolo weights to Darknet ',
        description='This script evaluates performances of model')

    parser.add_argument('--model_path', 
        help='Keras yolo weigthts model path (.h5)',
        default='../../model_data/Suspect/trained_weights_final_yolo_tiny.h5')

    parser.add_argument('--config_path', 
        help='Config yolo path (.cfg)',
        default='../../model_data/Suspect/yolov3-tiny-suspect.cfg')

    parser.add_argument('--yolo_anchors', 
        help='Config yolo path (.txt)',
        default='../../model_data/tiny_yolo_anchors.txt')

    parser.add_argument('--classes_path', 
        help='Classes path (.txt)',
        default='../../model_data/Suspect/classes_suspect.txt')

    parser.add_argument('--output_weights_path', 
        help='Darknet weights output path (.weights)',
        default='../../model_data/Suspect/yolov3-tiny-suspect.weights')

    args = parser.parse_args()

    model_path = args.model_path
    config_path = args.config_path
    yolo_anchors = args.yolo_anchors
    classes_path = args.classes_path
    weight_file = open(args.output_weights_path, 'wb')

    yoloobj = YOLO(model_path, yolo_anchors, classes_path)
    model = yoloobj.yolo_model
    main()