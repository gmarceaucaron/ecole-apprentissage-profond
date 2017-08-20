# Source: https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne

from utils import gen_data
from rnn_Lasagne import *



def run_recurrent(
    # Min/max sequence length
    min_length = 50,
    max_length = 55,
    # Number of units in the hidden (recurrent) layern
    n_hidden = 100,
    # Number of training sequences in each batch
    batch_size = 100,
    # Optimization learning rate
    learning_rate = .001,
    # All gradients above this will be clipped
    grad_clipping = 100.0,
    # How often should we check the output?
    epoch_size = 100,
    # Number of epochs to train the net
    n_epochs = 10,
    model_name='single_layer_rnn'
    ):

    #############
    # Load Data #
    #############
    print("Loading data...")
    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data(min_length=min_length, 
                                      max_length=max_length,
                                      batch_size=batch_size
                             )
    
    ############################################
    # allocate symbolic variables for the data #
    ############################################
    target_var = T.vector('targets')

    ####################################################
    # BUILD MODEL (The model is a function in Lasagne) #
    ####################################################
    print('... building the model')
    model = None

    if model_name == 'single_layer_rnn':
        model = single_layer_rnn(shape=(batch_size, max_length, 2),
                                 n_hidden=n_hidden,
                                 grad_clipping=grad_clipping
                )
    elif model_name == 'two_layer_rnn':
        model = two_layer_rnn(
                    shape=(batch_size, max_length, 2),
                    n_hidden=n_hidden,
                    grad_clipping=grad_clipping
                )
    elif model_name == 'single_layer_bidirectional_rnn':
        model = single_layer_bidirectional_rnn(
                    shape=(batch_size, max_length, 2),
                    n_hidden=n_hidden,
                    grad_clipping=grad_clipping
                )
    elif model_name == 'two_layer_bidirectional_rnn':
        model = two_layer_bidirectional_rnn(
                    shape=(batch_size, max_length, 2),
                    n_hidden=n_hidden,
                    grad_clipping=grad_clipping
                )
    elif model_name == 'single_layer_lstm':
        model = single_layer_lstm(
                    shape=(batch_size, max_length, 2),
                    n_hidden=n_hidden,
                    grad_clipping=grad_clipping
                )
    elif model_name == 'single_layer_bidirectional_lstm':
        model = single_layer_bidirectional_lstm(
                    shape=(batch_size, max_length, 2),
                    n_hidden=n_hidden,
                    grad_clipping=grad_clipping
                )
    
    #####################
    # Training Function #
    #####################
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(model.l_out)

    # The network output will have shape (batch_size, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    prediction = network_output.flatten()
    # Our cost will be mean-squared error
    loss = T.mean((prediction - target_var)**2)

    # Retrieve all parameters from the network
    params = lasagne.layers.get_all_params(model.l_out)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=learning_rate)
    # Theano functions for training and computing loss

    train_model = theano.function(
        inputs=[model.l_in.input_var, target_var, model.l_mask.input_var],
        outputs=loss,
        updates=updates,
        name='train'
    )

    ##################################
    # Validation & Testing Functions #
    ##################################
    validate_model = theano.function(
        inputs=[model.l_in.input_var, target_var, model.l_mask.input_var],
        outputs=loss,
        name='validate'
    )


    ###############
    # TRAIN MODEL #
    ###############
    print('... Starting training')
    try:
        for epoch in range(n_epochs):
            for _ in range(epoch_size):
                X, y, m = gen_data(min_length=min_length,
                                   max_length=max_length,
                                   batch_size=batch_size
                                   )
                train_model(X, y, m)
            val_err = validate_model(X_val, y_val, mask_val)
            print("Epoch {} validation error = {}".format(epoch, val_err))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    """
    Choose a model:
        single_layer_rnn
        two_layer_rnn # ERROR
        single_layer_bidirectional_rnn
        two_layer_bidirectional_rnn
        single_layer_lstm
        single_layer_bidirectional_lstm
    """
    run_recurrent(model_name='two_layer_rnn')

