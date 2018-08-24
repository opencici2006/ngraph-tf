import tensorflow as tf
import argparse
import numpy as np
import ngraph
import json
import os

def load_graph(frozen_graph_filename, select_device):
    """Load a tensorflow frozen graph.

    Loads the protobuf file from the disk and parse it to retrieve the
    unserialized graph_def.

    Args:
        frozen_graph_filename: The location of the tensorflow frozen graph file.
        select_device: CPU or NGRAPH.

    Returns:
        A tensorflow graph from the frozen model.
    """
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        with tf.device('/job:localhost/replica:0/task:0/device:' + select_device + ':0'):
            tf.import_graph_def(graph_def)
    return graph


def calculate_output(graph, input_example, input_tensor_name, output_tensor_name):
    """Calculate the output of the graph given the input.

    Get the tensors based on the input and output name from the graph,
    then feed the input_example to the graph and retrieves the output vector.

    Args:
	graph: A tensorflow graph or NGRAPH graph.
	input_example: Random generated input or actual image.
	input_tensor_name: Input tensor name in the frozen graph. 
	output_tensor_name: Output tensor name in the frozen graph.

    Returns:
	The output vector obtained from running the input_example through the graph.
    """
    input_placeholder = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        #log_device_placement=True,
        inter_op_parallelism_threads=1
    )

    with tf.Session(graph=graph, config=config) as sess:
        output_tensor = sess.run(output_tensor, feed_dict={
                                 input_placeholder: input_example})
        #print (output_tensor)
    	return output_tensor


def calculate_l1_norm(ngraph_output, tf_output):
    """Calculate the L1 Loss between vectors.
    
    Calculate the L1 Loss between the NGRAPH and tensorflow output vectors.

    Args:
	ngraph_output: The output vector generated from NGRAPH graph.
	tf_output: The output vector generated from tensorflow graph.

    Returns:
	A number that is the L1 loss between the vectors.

    Raises:
	Exception: If the dimension of the two vectors mismatch.
    """
    if(ngraph_output.shape != tf_output.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    
    ngraph_output_squeezed = np.squeeze(ngraph_output)
    tf_output_squeezed = np.squeeze(tf_output)
    
    ngraph_output_flatten = ngraph_output_squeezed.flatten()
    tf_output_flatten = tf_output_squeezed.flatten() 

    '''
    if(ngraph_output_squeezed.shape != tf_output_squeezed.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    '''

    factor = np.prod(ngraph_output_squeezed.shape)
    #return np.linalg.norm(ngraph_output_squeezed - tf_output_squeezed) / (factor)
    #return np.sum(np.abs(ngraph_output_squeezed - tf_output_squeezed), axis=0)
    return np.sum(np.abs(ngraph_output_flatten - tf_output_flatten), axis=0) / factor

def calculate_l2_norm(ngraph_output, tf_output):
    """Calculate the L2 Loss between vectors.
    
    Calculate the L2 Loss between the NGRAPH and tensorflow output vectors.

    Args:
	ngraph_output: The output vector generated from NGRAPH graph.
	tf_output: The output vector generated from tensorflow graph.

    Returns:
	A number that is the L2 loss between the vectors.

    Raises:
	Exception: If the dimension of the two vectors mismatch.
    """
    if(ngraph_output.shape != tf_output.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    
    ngraph_output_squeezed = np.squeeze(ngraph_output)
    tf_output_squeezed = np.squeeze(tf_output)

    ngraph_output_flatten = ngraph_output_squeezed.flatten()
    tf_output_flatten = tf_output_squeezed.flatten() 
    if(ngraph_output_squeezed.shape != tf_output_squeezed.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    factor = np.prod(ngraph_output_squeezed.shape)
    #return np.linalg.norm(abs(ngraph_output_squeezed - tf_output_squeezed))**2 / factor
    #return np.sum(np.dot(np.abs(ngraph_output_squeezed - tf_output_squeezed), np.abs(ngraph_output_squeezed - tf_output_squeezed)))
    return np.sum(np.dot(np.abs(ngraph_output_flatten - tf_output_flatten), np.abs(ngraph_output_flatten - tf_output_flatten)))/factor

def calculate_inf_norm(ngraph_output, tf_output):
    """Calculate the L2 Loss between vectors.
    
    Calculate the L2 Loss between the NGRAPH and tensorflow output vectors.

    Args:
	ngraph_output: The output vector generated from NGRAPH graph.
	tf_output: The output vector generated from tensorflow graph.

    Returns:
	A number that is the L2 loss between the vectors.

    Raises:
	Exception: If the dimension of the two vectors mismatch.
    """
    if(ngraph_output.shape != tf_output.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    
    ngraph_output_squeezed = np.squeeze(ngraph_output)
    tf_output_squeezed = np.squeeze(tf_output)

    ngraph_output_flatten = ngraph_output_squeezed.flatten()
    tf_output_flatten = tf_output_squeezed.flatten() 

    if(ngraph_output_squeezed.shape != tf_output_squeezed.shape):
        raise Exception('ngraph output and tf output dimension mismatch')
    factor = np.prod(ngraph_output_squeezed.shape)

    #return np.linalg.norm(abs(ngraph_output_squeezed - tf_output_squeezed))**2 / factor
    #return np.sum(np.dot(np.abs(ngraph_output_squeezed - tf_output_squeezed), np.abs(ngraph_output_squeezed - tf_output_squeezed)))
    #return np.sum(np.dot(np.abs(ngraph_output_flatten - tf_output_flatten), np.abs(ngraph_output_flatten - tf_output_flatten)))/factor
    return np.linalg.norm((ngraph_output_flatten - tf_output_flatten),np.inf)

def parse_json():
    """
	Parse the user input json file.
    """
    global frozen_graph_location
    global input_tensor_name
    global output_tensor_name
    global l1_loss_threshold
    global l2_loss_threshold
    global input_dimension

    with open(os.path.abspath(args.json_file)) as f:
        parsed_json = json.load(f)
        frozen_graph_location = parsed_json['frozen_graph_location']
        input_tensor_name = parsed_json['input_tensor_name']
        output_tensor_name = parsed_json['output_tensor_name']
        l1_loss_threshold = parsed_json['l1_loss_threshold']
        l2_loss_threshold = parsed_json['l2_loss_threshold']
        input_dimension = parsed_json['input_dimension']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default="./example.json",
                        type=str,help="Model details in json format")
    args = parser.parse_args()

    parse_json()

    # TODO: How to check placement is actually on NGRAPH
   
    # Generate random input based on input_dimension 
    #np.random.seed(0)
    np.random.seed(100)
    random_input = np.random.rand(1, input_dimension)

    # Run the model on ngraph
    ngraph_graph = load_graph(frozen_graph_location, "NGRAPH")
    result_ngraph = calculate_output(
        ngraph_graph, random_input, input_tensor_name, output_tensor_name)

    # Run the model on tensorflow
    tf_graph = load_graph(frozen_graph_location, "CPU")
    result_tf_graph = calculate_output(
        tf_graph, random_input, input_tensor_name, output_tensor_name)

    l1_loss = calculate_l1_norm(result_ngraph, result_tf_graph)
    l2_loss = calculate_l2_norm(result_ngraph, result_tf_graph)
    inf_norm = calculate_inf_norm(result_ngraph, result_tf_graph)

    print ("The inf norm is ", inf_norm)
    print ("l1 loss is ", l1_loss)
    print ("l2 loss is ", l2_loss)

    if l1_loss > l1_loss_threshold:
        print ("The L1 Loss %f is greater than the threshold %f " %
               (l1_loss, l1_loss_threshold))

    if l2_loss > l2_loss_threshold:
        print ("The L2 Loss %f is greater than the threshold %f " %
               (l2_loss, l2_loss_threshold))

