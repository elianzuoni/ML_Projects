"""
This file contains functions useful for testing, either "manually" (i.e. by feeding a manually-trained predictor and a manually-extracted testing dataset to an assesser), or "automatically" (i.e. via cross validation, which does the splitting, the training, and the testing).  
A different metric is used for the test loss than for the train loss (i.e. the one minimised by the training): for example, when the classifier is being assessed,it is the fraction of misclassified points, which more closely reflects the actual goodness of fit.  
"""

# -*- coding: utf-8 -*-
import numpy as np


##### FUNCTIONS COMPUTING TEST LOSS


def compute_mse(a, b):
    """ Returns the mean square difference between a and b (continuous values). """
    return (1 / len(a)) * np.dot((a-b).T, a-b)


def compute_nhd(a, b):
    """ Returns the normalised Hamming distance between a and b (binary values). """
    return (1 / len(a)) * np.sum(np.where(a != b, 1, 0))


##### GENERIC IMPLEMENTATION OF PREDICTOR ASSESSER


def assess_predictor(y_test, tx_test, predictor, compute_test_loss):
    """ Returns the loss (specified by "compute_test_loss") between "y_test" and the prediction on "tx_test" """
    return compute_test_loss(y_test, predictor(tx_test))


##### ACTUAL PREDICTOR ASSESSERS


def assess_regressor_mse(y_test, tx_test, regressor):
    """ Assesses the provided regressor with MSE loss function """
    return assess_predictor(y_test, tx_test, regressor, compute_mse)


def assess_classifier_nhd(y_test, tx_test, classifier):
    """ Assesses the provided classifier with NHD loss function """
    return assess_predictor(y_test, tx_test, classifier, compute_nhd)


##### CROSS VALIDATION


def cross_validation(y, tx, trainer, hyper_params, threshold, k_fold, to_assess="regressor"):
    """ Returns the average of the test loss and the train loss for this particular choice of "tx" (i.e. the way to
    pre-process data, like feature expansion) and "hyper_params" (extra parameters to the "trainer" function).
    Splits the data in "k_fold" equal parts, using each one once for testing, and "k_fold-1" times for training. """
    # Build index blocks
    N = y.shape[0]
    block_size = int(N / k_fold)
    indices = np.random.permutation(N)
    index_blocks = [indices[k*block_size: (k+1)*block_size] for k in range(k_fold)]
    
    avg_test_loss = 0
    avg_train_loss = 0
    for k in range(k_fold):
        # Get k-th block for testing, and remaining for training
        tx_test = tx[index_blocks[k], :]
        y_test = y[index_blocks[k]]
        # Initialisation to let the dimensions match
        tx_train = np.empty((1, tx.shape[1]))
        y_train = np.empty((1,))
        for j in range(k_fold):
            if j != k:
                # Concatenate along rows
                tx_train = np.concatenate((tx_train, tx[index_blocks[j], :]), axis=0)
                y_train = np.concatenate((y_train, y[index_blocks[j]]), axis=0)
        # Remove first row (initialisation)
        tx_train = tx_train[1:, :]
        y_train = y_train[1:]
                
        # Train model
        w, train_loss, regressor, classifier = trainer(y_train, tx_train, *hyper_params, threshold) # hyper_params is a tuple
        # Get test loss
        test_loss = 0
        if to_assess == "regressor":
            test_loss = assess_regressor_mse(y_test, tx_test, regressor)
        else:
            test_loss = assess_classifier_nhd(y_test, tx_test, classifier)
        
        # Update score
        avg_test_loss += test_loss
        avg_train_loss += train_loss
    
    # Return average losses
    return avg_test_loss/k_fold, avg_train_loss/k_fold
