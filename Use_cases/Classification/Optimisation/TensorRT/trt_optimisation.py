import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import argparse

from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser(
    prog='Classification - TensorRT',
    description='This script optimize TensorFlow model')

parser.add_argument('--pb_model_path', 
    help='Pb model\'s path (.pb)',
    default='tf_model/modelfire.pb')

parser.add_argument('--trt_output_path', 
    help='Trt model\'s path (.pb)',
    default='trt_graph_FP32.pb')

args = parser.parse_args()


with gfile.FastGFile(args.pb_model_path, "rb") as f:
    frozen_graph = tf.GraphDef()
    frozen_graph.ParseFromString(f.read())

k_outputs = ['dense_2/Softmax']

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=k_outputs,
    max_batch_size=1,  # CHANGE HERE
    max_workspace_size_bytes=1<<30, # CHANGE HERE
    precision_mode='FP32' # CHANGE HERE
)

tf.train.write_graph(trt_graph, "trt_model", args.trt_output_path, as_text=False)


#print("OK")

# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)
