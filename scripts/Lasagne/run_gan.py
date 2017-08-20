# Source (DCGAN): https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
# Source (WGAN): https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
# Source (LSGAN): https://gist.github.com/f0k/9b0bb51040719eeafec7eba473a9e79b

from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

from load_mnist_data import load_mnist_data
from utils import iterate_minibatches

############################
# Build Generator Function #
############################
def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
    try:
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    except ImportError:
        raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                          "version: http://lasagne.readthedocs.io/en/latest/"
                          "user/installation.html#bleeding-edge-version")
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return layer

#########################
# Build Critic Function #
#########################
def build_critic(input_var=None, model_name='wgan'):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))

    # output layer
    if model_name == 'dcgan':
        layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    elif model_name == 'wgan':
    	layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    elif model_name == 'lsgan':
        layer = DenseLayer(layer, 1, nonlinearity=None)

    print ("critic output:", layer.output_shape)
    return layer

def run_gan(
	num_epochs=1000,
	epochsize=100,
	batchsize=64,
	initial_eta=5e-5,
	clip=0.01,
	model_name='wgan',
	optimizer_name = 'rmsprop'
	):

	#############
    # Load Data #
    #############
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    input_shape = X_train[0].shape
    shape = (None, input_shape[0], input_shape[1], input_shape[2])

    ############################################
    # allocate symbolic variables for the data #
    ############################################
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

	####################################################
    # BUILD MODEL (The model is a function in Lasagne) #
    ####################################################
    print('... building the model')
    generator = build_generator(noise_var)
    critic = build_critic(input_var, model_name=model_name)

    #####################
    # Training Function #
    #####################

    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(critic,
            lasagne.layers.get_output(generator))

    # Create loss expressions
    if model_name == 'dcgan':
        generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
        critic_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
	            + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()
    elif model_name == 'wgan':
        generator_loss = -fake_out.mean()
        critic_loss = fake_out.mean() - real_out.mean()
    elif model_name == 'lsgan':
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        generator_loss = lasagne.objectives.squared_error(fake_out, c).mean()
        critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
	                   lasagne.objectives.squared_error(fake_out, a).mean())

	# Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))

    if optimizer_name == 'rmsprop':
        generator_updates = lasagne.updates.rmsprop(
            generator_loss, generator_params, learning_rate=eta)
        critic_updates = lasagne.updates.rmsprop(
            critic_loss, critic_params, learning_rate=eta)
    elif optimizer_name == 'adam':
        generator_updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5)
        critic_updates = lasagne.updates.adam(
            critic_loss, critic_params, learning_rate=eta, beta1=0.5)

    if clip:
		 # Clip critic parameters in a limited range around zero (except biases)
	    for param in lasagne.layers.get_all_params(critic, trainable=True,
	                                               regularizable=True):
	        critic_updates[param] = T.clip(critic_updates[param], -clip, clip)

	# Instantiate a symbolic noise generator to use for training
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_loss,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function([input_var], critic_loss,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    ###############
    # TRAIN MODEL #
    ###############
    print('... Starting training')
    # We create an infinite supply of batches (as an iterable generator):
    batches = iterate_minibatches(X_train, y_train, batchsize, shuffle=True)
    # We iterate over epochs:
    generator_updates = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        # In each epoch, we do `epochsize` generator and critic updates.
        critic_losses = []
        generator_losses = []
        for _ in range(epochsize):
            inputs, targets = next(batches)
            critic_losses.append(critic_train_fn(inputs))
            generator_losses.append(generator_train_fn())

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  generator loss: {}".format(np.mean(generator_losses)))
        print("  critic loss:    {}".format(np.mean(critic_losses)))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            plt.imsave('lsgan_mnist_samples.png',
                       (samples.reshape(6, 7, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(6*28, 7*28)),
                       cmap='gray')

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    ###################
    # SAVE/LOAD MODEL #
    ###################
    # Optionally, you could now dump the network weights to a file like this:
    np.savez('lsgan_mnist_gen.npz', *lasagne.layers.get_all_param_values(generator))
    np.savez('lsgan_mnist_crit.npz', *lasagne.layers.get_all_param_values(critic))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
	"""
	Choose a model (each model has slight modifications from the paper):
		dcgan (Deep Convolutional Generative Adversarial Networks)
		wgan (Wassrestein Generative Adversarial Network)
		lsgan (Least Squared Generative Adversarial Ntwork)
	Choose an optimiation method:
		rmsprop
		adam
	"""
	run_gan(model_name='lsgan', optimizer_name='adam')


