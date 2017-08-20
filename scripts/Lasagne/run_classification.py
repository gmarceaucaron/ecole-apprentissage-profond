"""
Source: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
Usage example employing Lasagne for digit recognition using the MNIST dataset.
A good Lasagne tutorial is here: https://github.com/craffel/Lasagne-tutorial/blob/master/examples/tutorial.ipynb
"""

from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from load_mnist_data import load_mnist_data
from utils import iterate_minibatches
from mlp_Lasagne import *
from cnn_Lasagne import *

def run_classification(n_epochs=5,
                       batch_size=64,
                       dataset='mnist.pkl.gz',
                       input_shape=(None,1,28,28), # CNN and LeNet5, this is MNIST dimensions                                                                                                                                                                                        
                       # or file path to load a saved file
                       model_name='single_layer_mlp',
    ):

    #############
    # Load Data #
    #############
    print("Loading data...")
    train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = load_mnist_data()

    ############################################
    # allocate symbolic variables for the data #
    ############################################
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    ####################################################
    # BUILD MODEL (The model is a function in Lasagne) #
    ####################################################
    print('... building the model')
    model = None

    # MLPs
    if model_name == 'single_layer_mlp':
        model = single_layer_mlp(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_mlp':
        model = two_layer_mlp(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_mlp_with_dropout':
        model = two_layer_mlp_with_dropout(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_mlp_with_batch_normalization':
        model = two_layer_mlp_with_batch_normalization(shape=input_shape, input_var=input_var)
    elif model_name == 'custom_mlp':
        model = custom_mlp(shape=input_shape, input_var=input_var)

    # CNNs
    elif model_name == 'two_layer_cnn':
        model = two_layer_cnn(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_cnn_with_dropout':
        model = two_layer_cnn_with_dropout(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_cnn_with_dropout_and_two_3x3_kernels':
        model = two_layer_cnn_with_dropout_and_two_3x3_kernels(shape=input_shape, input_var=input_var)
    elif model_name == 'two_layer_cnn_with_batch_normalization':    
        model = two_layer_cnn_with_batch_normalization(shape=input_shape, input_var=input_var)

    #####################
    # Training Function #
    #####################
    # Create a loss expression for training
    train_prediction = lasagne.layers.get_output(model)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var).mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(model, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_model = theano.function(
        inputs=[input_var, target_var],
        outputs=loss,
        updates=updates,
        name='train'
    )

    ##################################
    # Validation & Testing Functions #
    ##################################
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(model, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # We also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    validate_model = theano.function(
        inputs=[input_var, target_var], 
        outputs=[test_loss, test_acc],
        name='validate'
    )
    
    test_model = theano.function(
        inputs=[input_var, target_var],
        outputs=[test_loss, test_acc],
        name='test'
    )

    ##################
    # Training Model #
    ##################
    print('... Starting training')

    for epoch in range(n_epochs):
        # In each epoch, we do a full pass over the training data to train:
        start_time = time.time()
        for batch in iterate_minibatches(train_set_x, train_set_y, batch_size, shuffle=True):
            inputs, targets = batch
            train_model(inputs, targets)

        # And another full pass over the training data for calculating error:
        train_err = 0
        train_acc = 0
        train_batches = 0
        for batch in iterate_minibatches(train_set_x, train_set_y, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = validate_model(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_set_x, val_set_y, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = validate_model(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


    ##############
    # TEST MODEL #
    ##############
    print('... Starting testing')

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test_set_x, test_set_y, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = test_model(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':
    """
    Choose a model name:
    # MLPs
    single_layer_mlp
    two_layer_mlp
    two_layer_mlp_with_dropout
    two_layer_mlp_with_batch_normalization
    custom_mlp
    # CNNs
    two_layer_cnn
    two_layer_cnn_with_dropout
    two_layer_cnn_with_dropout_and_two_3x3_kernels
    two_layer_cnn_with_batch_normalization
    """
    run_classification(model_name='two_layer_cnn_with_dropout')
