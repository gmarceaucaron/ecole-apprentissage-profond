import lasagne

# 1600u FC - Output - Output
def single_layer_mlp(shape=None, input_var=None):

    # For MNIST, Input layer specifies the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

    # Add a fully-connected layer of 1600 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=1600,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

# 800u FC - 800u FC - Output
def two_layer_mlp(shape=None, input_var=None):

    l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    
    l_hid1 = lasagne.layers.DenseLayer(
                l_in, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2 = lasagne.layers.DenseLayer(
                l_hid1, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(
                l_hid2, num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# 800u FC w/dropout - 800u FC w/dropout - Output
# Dropout, 20% on input, and 50 % on hidden layers.
def two_layer_mlp_with_dropout(shape=None, input_var=None):
    
    l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# (BN - 800u FC) - (BN - 800u FC) - Output
# Dropout, 20% on input, and 50 % on hidden layers.
def two_layer_mlp_with_batch_normalization(shape=None, input_var=None):
    
    l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid1_BN = lasagne.layers.batch_norm(l_hid1)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_BN, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_BN = lasagne.layers.batch_norm(l_hid2)

    l_out = lasagne.layers.DenseLayer(
            l_hid2_BN, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def custom_mlp(shape=None, input_var=None):

    l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
    
    pass
    ### YOUR CODE HERE ###
    #network = None
    ###
    
    l_out = lasagne.layers.DenseLayer(
            network, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# Source: https://github.com/craffel/Lasagne-tutorial/blob/master/examples/mnist.py
def custom_mlp_2(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network