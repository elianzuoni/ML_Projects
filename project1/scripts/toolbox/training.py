"""
This file contains training functions, adapting those in implementations.py, offering an intuitive and homogeneous interface.  
The functions are named "train_(reg/unreg)_<cost>_<method>": the name specifies the cost function ("ls" for Least-Squares, "log" for Logistic), whether or not it is regularised, and the method used to optimise it (GD, SGD, or NE, for Normal Equations).  
These functions return (w, train_loss, regressor, classifier), where w is the optimal weight vector, train_loss is the loss on the training dataset, regressor is a function (parametrised by w) mapping datapoints to continuous predictions, and classifier is the function (parametrised by the regressor and a threshold) mapping datapoints to predictions in {0, 1}.
"""

# -*- coding: utf-8 -*-
import numpy as np
from .implementations import *


##### BUILDING BLOCKS


def get_linear_regressor(w):
    """ Returns the linear regressor function, given its weights. """
    return lambda tx : np.dot(tx, w)


def get_logistic_regressor(w):
    """ Returns the logistic regressor function, given its weights. """
    return lambda tx : sigmoid(np.dot(tx, w))


def get_classifier(regressor, threshold):
    """ Returns the classifier, given the regressor and the threshold. """
    return lambda tx : np.where(regressor(tx) < threshold, 0, 1)


##### GENERIC IMPLEMENTATION OF TRAINING FUNCTION


def train(learn_args, learn, get_regressor, threshold):
    """ Learns the model by supplying the tuple "learn_args" to the learning function "learn".
    Returns the model (w, loss) and the corresponding regressor and classifier. """
    # Learn the model
    w, loss = learn(*learn_args)
    
    # Get the regressor
    regressor = get_regressor(w)
    # Get the classifier
    classifier = get_classifier(regressor, threshold)
    
    return w, loss, regressor, classifier


##### CONCRETE TRAINING FUNCTIONS: LEAST SQUARES


def train_reg_ls_GD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    """ Training for L2-regularised Least-Squares loss with Gradient Descent """
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma)
    return train(learn_args, reg_least_squares_GD, get_linear_regressor, threshold)


def train_reg_ls_SGD(y, tx, lambda_, initial_w, max_iters, batch_size, gamma, threshold):
    """ Training for L2-regularised Least-Squares loss with Stochastic Gradient Descent """
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma, batch_size)
    return train(learn_args, reg_least_squares_SGD, get_linear_regressor, threshold)


def train_reg_ls_NE(y, tx, lambda_, threshold):
    """ Training for L2-regularised Least-Squares loss with Normal Equations """
    learn_args = (y, tx, lambda_)
    return train(learn_args, ridge_regression, get_linear_regressor, threshold)


def train_unreg_ls_GD(y, tx, initial_w, max_iters, gamma, threshold):
    """ Training for unregularised Least-Squares loss with Gradient Descent """
    learn_args = (y, tx, initial_w, max_iters, gamma)
    return train(learn_args, least_squares_GD, get_linear_regressor, threshold)


def train_unreg_ls_SGD(y, tx, initial_w, max_iters, batch_size, gamma, threshold):
    """ Training for unregularised Least-Squares loss with Stochastic Gradient Descent """
    learn_args = (y, tx, initial_w, max_iters, gamma, batch_size)
    return train(learn_args, least_squares_SGD, get_linear_regressor, threshold)


def train_unreg_ls_NE(y, tx, threshold):
    """ Training for unregularised Least-Squares loss with Normal Equations """
    learn_args = (y, tx)
    return train(learn_args, least_squares, get_linear_regressor, threshold)


##### CONCRETE TRAINING FUNCTIONS: LOGISTIC


def train_reg_log_GD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    """ Training for L2-regularised (and normalised) Logistic loss with Gradient Descent """
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma, True)
    return train(learn_args, reg_logistic_regression, get_linear_regressor, threshold)


def train_reg_log_SGD(y, tx, lambda_, initial_w, max_iters, batch_size, gamma, threshold):
    """ Training for L2-regularised (and normalised) Logistic loss with Stochastic Gradient Descent """
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma, batch_size, True)
    return train(learn_args, reg_logistic_regression_SGD, get_linear_regressor, threshold)


def train_unreg_log_GD(y, tx, initial_w, max_iters, gamma, threshold):
    """ Training for unregularised (and normalised) Logistic loss with Gradient Descent """
    learn_args = (y, tx, initial_w, max_iters, gamma, True)
    return train(learn_args, logistic_regression, get_linear_regressor, threshold)


def train_unreg_log_SGD(y, tx, initial_w, max_iters, batch_size, gamma, threshold):
    """ Training for unregularised (and normalised) Logistic loss with Stochastic Gradient Descent """
    learn_args = (y, tx, initial_w, max_iters, gamma, batch_size, True)
    return train(learn_args, logistic_regression_SGD, get_linear_regressor, threshold)