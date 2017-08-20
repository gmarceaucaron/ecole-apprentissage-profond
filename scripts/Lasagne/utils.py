from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

# Batch iterator: This is just a simple helper function iterating over 
# training data in mini-batches of a particular size, optionally in random order. 
# It assumes data is available as numpy arrays. For big datasets, 
# you could load numpy arrays as memory-mapped files (np.load(..., mmap_mode='r')), 
# or write your own custom data iteration function. For small datasets, 
# you can also copy them to GPU at once for slightly improved performance. 
# This would involve several changes in the main program, though, 
# and is not demonstrated here. Notice that this function returns 
# only mini-batches of size batchsize. If the size of the data is not a 
# multiple of batchsize, it will not return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #assert len(inputs) == len(targets)
    #print(theano.tensor.shape(inputs))
    #print(theano.tensor.shape(targets))
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def gen_data(min_length, max_length, batch_size):
    # Source: # Source: https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    batch_size : int
        Number of samples in the batch.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (batch_size, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (batch_size,).
    mask : np.ndarray
        A binary matrix of shape (batch_size, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.
    References
    ----------
    [1] Hochreiter, Sepp, and Jurgen Schmidhuber. "Long short-term memory." 
    Neural computation 9.8 (1997): 1735-1780.
    [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(batch_size, max_length, 1)),
                        np.zeros((batch_size, max_length, 1))],
                       axis=-1)
    mask = np.zeros((batch_size, max_length))
    y = np.zeros((batch_size,))
    # Compute masks and correct values
    for n in range(batch_size):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))
