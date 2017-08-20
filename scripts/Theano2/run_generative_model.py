from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import load_data
from utils import tile_raster_images
from AutoEncoder import AutoEncoder
from CA import CA
from DA import DA
from RBM import RBM
from sample_RBM import sample_RBM

# Must (conda install PIL) for Image
try:
    import PIL.Image as Image
except ImportError:
    import Image


def run_generative_model(learning_rate=0.1, 
                         dataset='mnist.pkl.gz',
                         n_epochs=5,
                         batch_size=20, 
                         display_step=1000,
                         n_visible=28*28, # MNIST Pixels
                         n_hidden=500,
                         corruption_level=0.3, # DA
                         contraction_level=0.1, # CA
                         k=5, # RBM
                         chains=10, # RBM
                         output_folder='Generative_plots',
                         img_shape=(28,28), # image shape of MNIST for tile_raster_images
                         model_name='AutoEncoder',
                         
):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing AutoEncoder

    :type n_epochs: int
    :param n_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """

    # numpy random generator
    rng = np.random.RandomState(123)
    # create a Theano random generator that gives symbolic random values
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    #############
    # Load Data #
    #############
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    ###################################
    # Calculate number of Minibatches #
    ###################################
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ############################################
    # allocate symbolic variables for the data #
    ############################################
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch

    x = T.matrix('x')  # the data is presented as rasterized images

    ###############
    # BUILD MODEL #
    ###############
    print('... building the model')

    if model_name == 'AutoEncoder':
        model = AutoEncoder(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=n_visible,
            n_hidden=n_hidden
        )
    elif model_name == 'DA':
        model = DA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=n_visible,
            n_hidden=n_hidden
        )
    elif model_name == 'CA':
        model = CA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=n_visible,
            n_hidden=n_hidden,
            batch_size=batch_size
        )
    elif model_name == 'RBM':
        model = RBM(
            input=x,
            numpy_rng=rng,
            theano_rng=theano_rng,
            n_visible=n_visible,
            n_hidden=n_hidden
        )

    #####################
    # Training Function #
    #####################
    # COST & UPDATES

    if model_name == 'AutoEncoder':
        cost, updates = model.get_cost_updates(
            learning_rate=learning_rate
        )

    elif model_name == 'DA':
        cost, updates = model.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )
    
    elif model_name == 'CA': 
        cost, updates = model.get_cost_updates(
            contraction_level=contraction_level,
            learning_rate=learning_rate
        )

    elif model_name == 'RBM':
        # initialize storage for the persistent chain (state = hidden layer of chain)
        persistent_chain = theano.shared(np.zeros(shape=(batch_size, model.n_hidden),
                                                     dtype=theano.config.floatX),
                                         borrow=True)
        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = model.get_cost_updates(learning_rate=learning_rate,
                                               persistent=persistent_chain,
                                               k=k)


    # TRAINING FUNCTION
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    plotting_time = 0.

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(n_epochs):
        minibatch_avg_cost = []
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost.append(train_model(minibatch_index))

            # iteration number
            iter = epoch * n_train_batches + minibatch_index
            if iter % display_step == 0:
                print('training @ iter = ', iter)

        print('Training epoch %d, cost ' % epoch, np.mean(minibatch_avg_cost, dtype='float64'))

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
        tile_raster_images(X=model.W.get_value(borrow=True).T,
                           img_shape=img_shape, tile_shape=(10, 10),
                           tile_spacing=(1, 1)))

        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    
    image = Image.fromarray(
        tile_raster_images(X=model.W.get_value(borrow=True).T,
                           img_shape=img_shape, tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    
    image.save('trained_filters.png')

    #################################
    #     Sampling from the Model   #
    #################################
    #if model_name == 'RBM':
    #    sample_RBM(model=model, test_set_x=test_set_x, chains=20)

    
    ####################
    # Change Directory #
    ####################
    os.chdir('../')


if __name__ == '__main__':
    """
    Choose a model name:
        AutoEncoder
        DA
        CA
        RBM
    """
    run_generative_model(model_name='CA')

