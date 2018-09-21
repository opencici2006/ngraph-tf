import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
import pdb


def load_frozen(graph_file):
    graphdef = graph_pb2.GraphDef()
    with open(graph_file, "r") as f:
        protobuf_str = f.read()
        text_format.Merge(protobuf_str, graphdef)
    return graphdef

gdef = load_frozen('/localdisk/sarkars/workspace1/tf_ngtf_1/ngraph-tf/examples/keras_resnet_scrubbed.pbtxt')
with tf.Session() as sess:
    tf.import_graph_def(gdef)
    pdb.set_trace()
