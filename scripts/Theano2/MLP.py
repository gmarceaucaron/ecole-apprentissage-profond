import theano
import theano.tensor as T


from LogisticRegression import LogisticRegression
from util_layers import HiddenLayer, negative_log_likelihood


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, numpy_rng, input, n_in, n_hidden, n_out):
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
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.initialize_variables()

        self.params = self.hiddenLayer.params + self.logistic_regression_layer.params

        ################
        ## Prediction ##
        ################
        self.y_pred = self.logistic_regression_layer.y_pred

    def initialize_variables(self):
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=self.numpy_rng,
            input=self.input,
            n_in=self.n_in,
            n_out=self.n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logistic_regression_layer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=self.n_hidden,
            n_out=self.n_out
        )

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
        self.L1 = L1_reg * ( abs(self.hiddenLayer.W).sum()
                             + abs(self.logistic_regression_layer.W).sum()
                            )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = L2_reg * ( (self.hiddenLayer.W ** 2).sum()
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
