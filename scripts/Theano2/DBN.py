"""
"""
from __future__ import print_function, division
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from LogisticRegression import LogisticRegression
from utils import load_data
from util_layers import HiddenLayer, negative_log_likelihood
from RBM import RBM


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_in=28*28,
        hidden_layers_sizes=[500, 500],
        n_out=10
    ):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        # instance variables
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.input = input
        self.n_in = n_in
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0
        self.n_out = n_out

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # instance variable needed for stacking models
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []

        self.initialize_variables()

        ################
        ## Prediction ##
        ################
        self.y_pred = self.logistic_regression_layer.y_pred

    def initialize_variables(self):
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=self.numpy_rng,
                            theano_rng=self.theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=self.hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logistic_regression_layer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_out
        )

        self.params.extend(self.logistic_regression_layer.params)

    ##################################
    # Training Step Helper Functions #
    ##################################

    def get_cost(self, y, L1_reg, L2_reg):
        # loss function (without regularization)
        self.loss = negative_log_likelihood( 
                        self.logistic_regression_layer.p_y_given_x, 
                        y)
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = L1_reg * (np.sum(
                                np.array(
                                    [abs(sigmoid_layer.W).sum() for sigmoid_layer in self.sigmoid_layers]
                                ))
                            + abs(self.logistic_regression_layer.W).sum()
                            )


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = L2_reg *  (np.sum(
                                    np.array(
                                        [(sigmoid_layer.W ** 2).sum() for sigmoid_layer in self.sigmoid_layers]
                                    ))
                                + (self.logistic_regression_layer.W ** 2).sum()
                                )
        
        self.cost = self.loss + self.L1 + self.L2_sqr
        return self.cost

    def get_updates(self, cost, learning_rate):
        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list grads
        grads = [T.grad(cost, param) for param in self.params]
    
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, grads)
        ]
        return updates
    
    def get_cost_updates(self, y, L1_reg, L2_reg, learning_rate):
        cost = self.get_cost(y, L1_reg, L2_reg)
        updates = self.get_updates(cost, learning_rate)
        return cost, updates

    ##############################################
    # Accessor Methods for training step outputs #
    ##############################################

    def errors(self, y):
        return self.logistic_regression_layer.errors(y)

    def get_latest_cost(self):
        return self.cost

    def get_loss(self):
        return self.loss

    def get_L1(self):
        return self.L1

    def get_L2_sqr(self):
        return self.L2_sqr

    ################
    # Pre-training #
    ################

    def pretraining_functions(self, x, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        #x = T.matrix('x')  # data, presented as rasterized images

        learning_rate = T.scalar('lr')  # learning rate to use

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate=learning_rate,
                                                 persistent=None,
                                                 k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(learning_rate,value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
