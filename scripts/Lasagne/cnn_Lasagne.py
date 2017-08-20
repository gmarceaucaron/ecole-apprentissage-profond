import lasagne

# (32x5x5 conv - MaxPool) - (32x5x5 conv - MaxPool) - FC - FC - Output
def two_layer_cnn(shape=None, input_var=None):

    network = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    
    # Another choice of initializaiton is: W=lasagne.init.HeNormal(gain='relu'))
    # Other arguments: Convolution type (full, same, or valid) and stride
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# (32x5x5 conv - MaxPool) - (32x5x5 conv - MaxPool) - FC w/dropout - FC w/dropout - Output
def two_layer_cnn_with_dropout(shape=None, input_var=None):

    network = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# (44x3x3 conv - 44x3x3 conv - MaxPool) - (44x3x3 conv - 44x3x3 conv - MaxPool) - FC w/dropout - FC w/dropout - Output
# Same area of visible space, but less parameters than previous layer.
# NOTE: A single 5x5x32 layer has 800 parameters. Two 3x3x44 layers have a combined 792 parameters.
# It's better to have number of filters be a power of 2 for parallelization. This is for demonstration purposes to keep the number # of parameters roughly equivalent.
def two_layer_cnn_with_dropout_and_two_3x3_kernels(shape=None, input_var=None):

    network = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

    # 2 conv layers, both with 44 3x3 kernels:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=44, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=44, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    network = lasagne.layers.Conv2DLayer(
            network, num_filters=44, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=44, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

# Note: In lasagne, the batchnorm must be applied to an incoming layer
# (BN - 32x5x5 conv - MaxPool) - (BN - 32x5x5 conv - MaxPool) - (BN - FC w/dropout) - (BN - FC w/dropout) - Output
def two_layer_cnn_with_batch_normalization(shape=None, input_var=None):

    network = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.batch_norm(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.DenseLayer(
            network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.batch_norm(network)

    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def custom_cnn(shape=None, input_var=None):

    network = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    
    pass
    ### YOUR CODE HERE ###
    #network = None
    ###
    
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out