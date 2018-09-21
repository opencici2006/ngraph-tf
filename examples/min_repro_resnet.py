import tensorflow as tf
import pdb, numpy as np
import ngraph

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


graph = load_graph('./tmp/frozen_model.pb')

for op in graph.get_operations():
    if op.type=='Placeholder':
        print(op.name)
    if op.type=='Softmax':
        print(op.name)

x = graph.get_tensor_by_name('prefix/module/hub_input/images:0')
y = graph.get_tensor_by_name('prefix/module/resnet_v2_50/predictions/Softmax:0')

with tf.Session(graph=graph) as sess:
    for i in range(10000):
        y_out = sess.run(y, feed_dict={x: np.random.rand(128,224,224,3)})
        print(i)