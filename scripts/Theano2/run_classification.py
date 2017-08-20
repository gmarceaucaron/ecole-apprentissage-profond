from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import ipdb

import numpy as np

import theano
import theano.tensor as T

from utils import load_data
from LogisticRegression import LogisticRegression
from MLP import MLP
from DeepMLP import DeepMLP
from CNN import CNN
from LeNet5 import LeNet5
from SdA import SdA
from DBN import DBN


def run_classification(pretrain_lr=0.001, # SdA and DBN
                       learning_rate=0.01, 
                       L1_reg=0.001,
                       L2_reg=0.0001,
                       pretraining_epochs=3, # SdA and DBN
                       n_epochs=5,
                       batch_size=64,
                       display_step=1000,
                       dataset='mnist.pkl.gz',
                       n_in=28*28, # mnist image shape
                       input_shape=(-1,1,28,28), # CNN and LeNet5, this is MNIST dimensions
                       n_out=10, # number of MNIST classes
                       n_hidden=1000, # (1-layer) MLP
                       hidden_layers_sizes=[500,500,500],
                       CNN_filter_size=20, # CNN
                       LeNet5_filter_sizes=[50,20], # LeNet5
                       corruption_levels=[0.1,0.2,0.3], # SdA
                       k=1, # DBN
                       # model_name can be the name of a model to create,
                       # or file path to load a saved file
                       model_name='LogisticRegression',
                       best_model_file_path='best_model.pkl'
):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    ######################
    # Instance Variables #
    ######################
    # instance variables to be used in some of the models below
    numpy_rng = np.random.RandomState(1234)

    #############
    # Load Data #
    #############
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    val_set_x, val_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    ###################################
    # Calculate number of Minibatches #
    ###################################
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_val_batches = val_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    ############################################
    # allocate symbolic variables for the data #
    ############################################
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    ###############
    # BUILD MODEL #
    ###############
    print('... building the model')
    model=None

    if model_name == 'LogisticRegression':
        model = LogisticRegression(
            input=x,
            n_in=n_in,
            n_out=n_out
        )
    elif model_name == 'MLP':
        model = MLP(
            numpy_rng=numpy_rng,
            input=x,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out
        )
    elif model_name == 'DeepMLP':
        model = DeepMLP(
            numpy_rng=numpy_rng,
            input=x,
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            n_out=n_out
        )
    elif model_name == 'CNN':
        model = CNN(
                numpy_rng=numpy_rng,
                input=x,
                input_shape=input_shape,
                filter_sizes=[CNN_filter_size],
                n_out=n_out,
                batch_size=batch_size
            )
    elif model_name == 'LeNet5':
        model = LeNet5(
            numpy_rng=numpy_rng,
            input=x,
            input_shape=input_shape,
            filter_sizes=LeNet5_filter_sizes,
            n_out=n_out,
            batch_size=batch_size
        )
    elif model_name == 'SdA':
        model = SdA(
            numpy_rng=numpy_rng,
            input=x,
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            n_out=n_out
        )
    elif model_name == 'DBN':
        model = DBN(
            numpy_rng=numpy_rng, 
            input=x,
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            n_out=n_out
        )
    # Assume the model_name is a path
    elif model_name != None:
        try:
            model = pickle.load(open(model_name))
        except:
            raise "Error! Model file path not valid."
    else:
        raise "Error! No model selected."

    #########################################
    # PRETRAINING THE MODEL (SdA, DBN Only) #
    #########################################
    if (model_name == 'SdA') or (model_name == 'DBN'):
        print('... starting pretraining')

        #########################
        # PreTraining Functions #
        #########################
        print('... getting the pretraining functions')

        if model_name == 'SdA':
            pretraining_fns = model.pretraining_functions(
                                x=x, # I had to move x here, instead of in the model, or there was an error.
                                train_set_x=train_set_x,
                                batch_size=batch_size) 
        
        elif model_name == 'DBN':
            pretraining_fns = model.pretraining_functions(
                                x=x, # I had to move x here, instead of in the model, or there was an error.
                                train_set_x=train_set_x,
                                batch_size=batch_size,
                                k=k)

        ##################
        # PRETRAIN MODEL #
        ##################
        print('... pre-training the model')
        start_time = timeit.default_timer()

        if model_name == 'SdA':
            corruption_levels = [.1, .2, .3]
        ## Pre-train layer-wise
        for i in range(model.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                cost = []
                for batch_index in range(n_train_batches):

                    if model_name == 'SdA':
                        cost.append(
                            pretraining_fns[i](index=batch_index,
                                               corruption=corruption_levels[i],
                                               lr=pretrain_lr)
                        )
                    elif model_name == 'DBN':
                        cost.append(
                            pretraining_fns[i](index=batch_index,
                                               lr=pretrain_lr)
                        )

                print('Pre-training layer %i, epoch %d, cost %f' % 
                      (i, epoch+1, np.mean(cost, dtype='float64'))
                )

        end_time = timeit.default_timer()

        print(('The pretraining code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

        print('...End of pre-training')
    
    #####################
    # Training Function #
    #####################
    cost, updates = model.get_cost_updates(
        y=y,
        L1_reg = L1_reg, 
        L2_reg = L2_reg,
        learning_rate=learning_rate
    )

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    
    train_model = theano.function(
        inputs=[index],
        outputs=model.get_latest_cost(),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train'
    )
    
    ##################################
    # Validation & Testing Functions #
    ##################################

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    validate_model = theano.function(
        inputs=[index],
        outputs=[model.errors(y), model.get_loss(), model.get_L1(), model.get_L2_sqr()],
        givens={
            x: val_set_x[index * batch_size:(index + 1) * batch_size],
            y: val_set_y[index * batch_size:(index + 1) * batch_size]
        },
        name='validate'
    )

    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        },
        name='test'
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10 * n_train_batches # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    minibatch_training_costs = []

    # go through training epochs
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in range(n_train_batches):

            #################
            # Training Step #
            #################
            latest_minibatch_training_cost = train_model(minibatch_index)
            minibatch_training_costs.append(latest_minibatch_training_cost)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % display_step == 0:
                print('training @ iter = ', iter)

            if (iter + 1) % validation_frequency == 0:

                #################
                # Training Loss #
                #################
                this_training_loss = np.mean(minibatch_training_costs, dtype='float64')

                print('latest average training loss: %f' % (this_training_loss))
                minibatch_training_costs = []

                ###################
                # Validation Loss #
                ###################
                validation_losses = [validate_model(i)[0] for i in range(n_val_batches)]
                this_validation_loss = np.mean(validation_losses, dtype='float64')

                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                        (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)
                )

                ########################
                # Validation Sublosses #
                ########################
                # Latest sublosses for our models include: unregularized loss, L1_norm, L2_norm
                unregularized_losses = [validate_model(i)[1] for i in range(n_val_batches)]
                this_unregularized_loss = np.mean(unregularized_losses, dtype='float64')
                L1_losses = [validate_model(i)[2] for i in range(n_val_batches)]
                this_L1_loss = np.mean(L1_losses, dtype='float64')
                L2_sqr_losses = [validate_model(i)[3] for i in range(n_val_batches)]
                this_L2_sqr_loss = np.mean(L2_sqr_losses, dtype='float64')
                print('latest total validation loss: %f' % (this_unregularized_loss + this_L1_loss + this_L2_sqr_loss) )
                print('latest unregularized loss: %f' % (this_unregularized_loss) )
                print('latest L1_norm: %f' % (this_L1_loss) )
                print('latest L2_norm: %f' % (this_L2_sqr_loss) )

                ###################
                # Save Best Model #
                ###################
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < (best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    ###################
                    # Test Best Model #
                    ###################
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses, dtype='float64')

                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % 
                            (epoch, minibatch_index + 1, n_train_batches, test_score * 100.)
                    )

                    ###################
                    # Sav Best Model #
                    ###################
                    with open(best_model_file_path, 'wb') as f:
                        pickle.dump(model, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def predict(dataset='mnist.pkl.gz', model_path='best_model.pkl'):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """
    num_predict = 10

    #############
    # Load Data #
    #############
    # This is currently hardcoded to handle the MNIST dataset.
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]

    ##########################
    # Data Values to Predict #
    ##########################
    test_set_x = test_set_x.get_value()
    prediction_set_x = test_set_x[:num_predict]

    ####################
    # Load Saved Model #
    ####################
    # load the saved model
    model = pickle.load(open(model_path))

    ######################
    # Predictor Function #
    ######################
    predict_model = theano.function(
        inputs=[model.input],
        outputs=model.y_pred)

    # We can test it on some examples from test test
    predicted_values = predict_model(prediction_set_x)

    ###########
    # Results #
    ###########
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

if __name__ == '__main__':
    """
    Choose a model name:
        LogisticRegression
        MLP
        DeepMLP
        CNN
        LeNet5
        SDA
        DBN
    """
    run_classification(model_name='LogisticRegression')
    print('\nPrediction:')
    predict()
 

