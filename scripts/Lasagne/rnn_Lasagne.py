# Source: http://lasagne.readthedocs.io/en/latest/modules/layers/recurrent.html

import lasagne

class single_layer_rnn(object):

    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100,  grad_clipping=100.0):
        # First, we build the network, starting with an input layer
        # Recurrent layers expect input of shape
        # (batch size, max sequence length, number of features)
        # By setting the first and second dimensions to None, we allow
        # arbitrary minibatch sizes with arbitrary sequence lengths.
        # The number of feature dimensions is 2, as described above.
        self.l_in = lasagne.layers.InputLayer(shape=shape,input_var=input_var)
        # The network also needs a way to provide a mask for each sequence.  We'll
        # use a separate input layer for that.  Since the mask only determines
        # which indices are part of the sequence for each batch entry, they are
        # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))

        # Setting a value for grad_clipping will clip the gradients in the layer
        # Setting only_return_final=True makes the layers only return their output
        # for the final time step, which is all we need for this task
        self.l_hidden = lasagne.layers.RecurrentLayer(self.l_in,
                                                      n_hidden,
                                                      mask_input=self.l_mask,
                                                      grad_clipping=grad_clipping,
                                                      W_in_to_hid=lasagne.init.HeUniform(),
                                                      W_hid_to_hid=lasagne.init.HeUniform(),
                                                      nonlinearity=lasagne.nonlinearities.tanh,
                                                      only_return_final=True
                                                      )

        self.l_out = lasagne.layers.DenseLayer(self.l_hidden,
                                               num_units=1,
                                               nonlinearity=lasagne.nonlinearities.tanh
                                               )
    
class two_layer_rnn(object):
    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100,  grad_clipping = 100.0):

        self.l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))
        
        self.l_hidden_1 = lasagne.layers.RecurrentLayer(
            self.l_in, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
        
        # Second hidden layer
        self.l_hidden_2 = lasagne.layers.RecurrentLayer(
            self.l_hidden_1, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
        
        self.l_out = lasagne.layers.DenseLayer(
            self.l_hidden_2, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

class single_layer_bidirectional_rnn(object):
    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100,  grad_clipping = 100.0):

        self.l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))
        
        # We're using a bidirectional network, which means we will combine two
        # RecurrentLayers, one with the backwards=True keyword argument.
        self.l_forward = lasagne.layers.RecurrentLayer(
            self.l_in, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, 
            only_return_final=True)
        self.l_backward = lasagne.layers.RecurrentLayer(
            self.l_in, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True, backwards=True)
        
        # Now, we'll concatenate the outputs to combine them.
        self.l_concat = lasagne.layers.ConcatLayer([self.l_forward, self.l_backward])

        self.l_out = lasagne.layers.DenseLayer(
            self.l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

class two_layer_bidirectional_rnn(object):
    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100,  grad_clipping = 100.0):

        self.l_in = lasagne.layers.InputLayer(shape=shape,input_var=input_var)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))
        
        self.l_forward_1 = lasagne.layers.RecurrentLayer(
            self.l_in, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, 
            only_return_final=True)
        self.l_backward_1 = lasagne.layers.RecurrentLayer(
            self.l_in, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True, backwards=True)
        
        # Second bidirectional hidde layer
        self.l_forward_2 = lasagne.layers.RecurrentLayer(
            self.l_forward_1, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh, 
            only_return_final=True)
        self.l_backward_2 = lasagne.layers.RecurrentLayer(
            self.l_backward_1, n_hidden, mask_input=self.l_mask, grad_clipping=grad_clipping,
            W_in_to_hid=lasagne.init.HeUniform(),
            W_hid_to_hid=lasagne.init.HeUniform(),
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True, backwards=True)
        
        self.l_concat = lasagne.layers.ConcatLayer([l_forward_2, l_backward_2])

        self.l_out = lasagne.layers.DenseLayer(
            self.l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)


class single_layer_lstm(object):
    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100,  grad_clipping = 100.0):

        self.l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))
        
        # All gates have initializers for the input-to-gate and hidden state-to-gate
        # weight matrices, the cell-to-gate weight vector, the bias vector, and the nonlinearity.
        # The convention is that gates use the standard sigmoid nonlinearity,
        # which is the default for the Gate class.
        self.gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), 
            W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))
        self.cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), 
            W_hid=lasagne.init.Orthogonal(),
            # Setting W_cell to None denotes that no cell connection will be used.
            W_cell=None,
            b=lasagne.init.Constant(0.),
            # By convention, the cell nonlinearity is tanh in an LSTM.
            nonlinearity=lasagne.nonlinearities.tanh)
        
        self.l_lstm = lasagne.layers.recurrent.LSTMLayer(
            self.l_in, n_hidden, mask_input=self.l_mask,
            # Here, we supply the gate parameters for each gate
            ingate=self.gate_parameters, forgetgate=self.gate_parameters,
            cell=self.cell_parameters, outgate=self.gate_parameters,
            # We'll learn the initialization and use gradient clipping
            learn_init=True, 
            grad_clipping=grad_clipping)
        
        self.l_out = lasagne.layers.DenseLayer(
            self.l_lstm, num_units=1, nonlinearity=lasagne.nonlinearities.tanh) 
    
class single_layer_bidirectional_lstm(object):
    def __init__(self, shape=(None, None, 2), input_var=None, n_hidden=100, grad_clipping = 100.0):
    
        self.l_in = lasagne.layers.InputLayer(shape=shape, input_var=input_var)
        self.l_mask = lasagne.layers.InputLayer(shape=(None, None))
        
        self.gate_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), 
            W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.))
        self.cell_parameters = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), 
            W_hid=lasagne.init.Orthogonal(),
            W_cell=None,
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh)
        
        self.l_lstm_forward = lasagne.layers.recurrent.LSTMLayer(
            self.l_in, n_hidden, mask_input=self.l_mask,
            ingate=self.gate_parameters, forgetgate=self.gate_parameters,
            cell=self.cell_parameters, outgate=self.gate_parameters,
            learn_init=True,
            grad_clipping=grad_clipping)
        
        # The "backwards" layer is the same as the first,
        # except that the backwards argument is set to True.
        self.l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
            self.l_in, n_hidden, ingate=self.gate_parameters,
            mask_input=self.l_mask, forgetgate=self.gate_parameters,
            cell=self.ell_parameters, outgate=self.gate_parameters,
            learn_init=True, grad_clipping=100., backwards=True)
        # We'll combine the forward and backward layer output by summing.
        # Merge layers take in lists of layers to merge as input.
        self.l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm, l_lstm_back])
        
        self.l_out = lasagne.layers.DenseLayer(
            self.l_sum, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

    
def hidden_recurrent_layer(layer_in, n_hidden, mask_input, grad_clipping=100., backwards=False, only_return_final=True):
    layer = lasagne.layers.RecurrentLayer(
        layer_in, n_hidden, mask_input=l_mask, 
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, 
        backwards=backwards,
        only_return_final=only_return_final)
    return layer

def cell_params():
    cell_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), 
        W_hid=lasagne.init.Orthogonal(),
        W_cell=None,
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh)
    return cell_parameters

def gate_params():
    gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), 
        W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))
    return gate_parameters