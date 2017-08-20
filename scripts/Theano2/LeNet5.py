import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from LogisticRegression import LogisticRegression
from utils import load_data
from util_layers import LeNetConvPoolLayer, HiddenLayer, negative_log_likelihood

class LeNet5(object):

    def __init__(self, input, image_shape, numpy_rng, filter_sizes, batch_size, n_out=10):

        # instance variables
        self.numpy_rng = numpy_rng
        self.input = input
        self.image_shape = image_shape
        self.filter_sizes = filter_sizes
        self.n_out = n_out
        self.batch_size = batch_size
        
        self.params = []

        self.initialize_variables()

        ################
        ## Prediction ##
        ################
        self.y_pred = self.logistic_regression_layer.y_pred

    def initialize_variables(self):

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.layer0_input = self.input.reshape((-1, self.image_shape[1], self.image_shape[2], self.image_shape[3]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, filter_sizes[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            rng = self.numpy_rng,
            input=self.layer0_input,
            image_shape=(-1, self.image_shape[1], self.image_shape[2], self.image_shape[3]),
            filter_shape=(self.filter_sizes[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        self.params.extend(self.layer0.params)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, filter_sizes[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            rng = self.numpy_rng,
            input=self.layer0.output,
            image_shape=(-1, self.filter_sizes[0], 12, 12),
            filter_shape=(self.filter_sizes[1], self.filter_sizes[0], 5, 5),
            poolsize=(2, 2)
        )
        self.params.extend(self.layer1.params)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, filter_sizes[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng=self.numpy_rng,
            input=self.layer2_input,
            n_in=self.filter_sizes[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )
        self.params.extend(self.layer2.params)

        # classify the values of the fully-connected sigmoidal layer
        self.logistic_regression_layer = LogisticRegression(input=self.layer2.output, n_in=500, n_out=self.n_out)
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
        self.L1 = L1_reg * ( abs(self.layer0.W).sum()
                             + abs(self.layer1.W).sum()
                             + abs(self.layer2.W).sum()
                             + abs(self.logistic_regression_layer.W).sum()
                            )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = L2_reg * ( (self.layer0.W ** 2).sum()
                                 + (self.layer1.W ** 2).sum()
                                 + (self.layer2.W ** 2).sum()
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