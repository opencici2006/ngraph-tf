import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#import ngraph
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2

with tf.Graph().as_default():

    module = hub.Module(
        "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")
    height, width = hub.get_expected_image_size(module)
    images = np.random.rand(
        32, height, width,
        3)  # A batch of images with shape [batch_size, height, width, 3].
    logits = module(images)  # Logits with shape [batch_size, num_classes].

    saver = tf.train.Saver()
    flag = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for i in range(10000):
            print(i)
            (sess.run(logits))
            if not flag:
                flag = True
                save_path = saver.save(sess, "./tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)
                tf.train.write_graph(sess.graph, './tmp1/',
                                     'tfhub_resnet.pbtxt')
                input_graph_path = './tmp1/tfhub_resnet.pbtxt'
                input_saver_def_path = ""
                input_binary = False
                output_node_names = ['fc1000/Softmax']
                restore_op_name = None
                filename_tensor_name = None
                output_graph_path = './tmp2'
                clear_devices = False
                input_meta_graph = './tmp/model.ckpt.meta'

                freeze_graph.freeze_graph(
                    input_graph_path,
                    input_saver_def_path,
                    input_binary,
                    save_path,
                    output_node_names,
                    restore_op_name,
                    filename_tensor_name,
                    output_graph_path,
                    clear_devices,
                    "",
                    "",
                    input_meta_graph,
                    checkpoint_version=saver_pb2.SaverDef.V2)
                #freeze_graph.freeze_graph('./tfhub_resnet.pbtxt', "",False,'./tmp/model.ckpt', "output/predictions", "save/restore_all",  "save/Const:0",'frozen.pb', True,"")
