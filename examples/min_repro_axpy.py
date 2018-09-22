import tensorflow as tf
import pdb, numpy as np
import ngraph
from google.protobuf import text_format


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        #graph_def.ParseFromString(f.read())
        text_format.Merge(f.read(), graph_def)

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


graph = load_graph('../test/test_axpy_const.pbtxt')

for op in graph.get_operations():
    print(op.name, op.type)

x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/y:0')
a = graph.get_tensor_by_name('prefix/add:0')
m = graph.get_tensor_by_name('prefix/mul:0')

with tf.Session(graph=graph) as sess:
    for i in range(100000000):
        y_out = sess.run([a, m],
                         feed_dict={
                             x: np.random.rand(2, 3),
                             y: np.random.rand(2, 3)
                         })
        #print(i)
