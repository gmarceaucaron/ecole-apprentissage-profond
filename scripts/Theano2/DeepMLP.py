import numpy as np

import theano
import theano.tensor as T


from LogisticRegression import LogisticRegression
from utils import load_data
from util_layers import HiddenLayer, negative_log_likelihood


class DeepMLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, numpy_rng, input, n_in, hidden_layers_sizes, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # instance variables
        self.numpy_rng = numpy_rng
        self.input = input
        self.n_in = n_in
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_layers = len(hidden_layers_sizes)
        self.n_out = n_out

        self.hidden_layers = []
        self.params = []

        self.initialize_variables()


        ################
        ## Prediction ##
        ################
        self.y_pred = self.logistic_regression_layer.y_pred

        

    def initialize_variables(self):

        for i in range(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.hidden_layers[-1].output

            hidden_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        activation=T.tanh)

            # add the layer to our list of layers
            self.hidden_layers.append(hidden_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(hidden_layer.params)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logistic_regression_layer = LogisticRegression(
            input=self.hidden_layers[-1].output,
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
                                    [abs(hidden_layer.W).sum() for hidden_layer in self.hidden_layers]
                                ))
                            + abs(self.logistic_regression_layer.W).sum()
                            )


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = L2_reg *  (np.sum(
                                    np.array(
                                        [(hidden_layer.W ** 2).sum() for hidden_layer in self.hidden_layers]
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
