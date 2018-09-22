# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np, os
from tensorflow.python.tools import freeze_graph
import tensorflow.keras.backend as K

#import ngraph

# A simple script to run inference and training on resnet 50


#https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        # https://github.com/onnx/tensorflow-onnx/issues/77
        # fix nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


model = ResNet50(weights='imagenet')

batch_size = 128
img = np.random.rand(batch_size, 224, 224, 3)
flag = False
for i in range(1000):
    preds = model.predict(preprocess_input(img))
    if not flag:
        import pdb
        pdb.set_trace()
        frozen_graph = freeze_session(
            K.get_session(),
            output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(
            frozen_graph,
            "/localdisk/sarkars/workspace1/tf_ngtf_1/ngraph-tf/examples",
            "keras_resnet_scrubbed.pbtxt",
            as_text=True)
        #tf.train.write_graph(tf.keras.backend.get_session().graph, '/localdisk/sarkars/workspace1/tf_ngtf_1/ngraph-tf/examples', 'keras_resnet.pbtxt')
        flag = True
        print("DONE")
print('Predicted:', decode_predictions(preds, top=3)[0])
#model.compile(tf.keras.optimizers.SGD(), loss='categorical_crossentropy')
#for i in range(1000):
#    preds = model.fit(preprocess_input(img), np.zeros((batch_size, 1000), dtype='float32'))
#print('Ran a train round')

#grep "Placeholder" -B 3 keras_resnet_scrubbed.pbtxt
#this tells us: input_1
