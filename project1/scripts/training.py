"""
This file contains training functions, adapting those in implementations.py, offering an intuitive and homogeneous interface.
The functions are named train_(reg/unreg)_<cost>_<method>, and they return (w, train_loss, regressor, classifier), where w is the
optimal weight vector, train_loss is the loss on the training dataset, regressor is the function (parametrised by w)
mapping datapoints to continuous outputs, and classifier is the function mapping datapoints to {0, 1} 
"""

# -*- coding: utf-8 -*-
import numpy as np
from implementations import *


##### BUILDING BLOCKS


""" Returns the linear regressor function, given its weights """
def get_linear_regressor(w):
    return lambda tx : np.dot(tx, w)


""" Returns the logistic regressor function, given its weights """
def get_logistic_regressor(w):
    return lambda tx : sigmoid(np.dot(tx, w))


""" Returns the classifier, given the regressor and the threshold """
def get_classifier(regressor, threshold):
    return lambda tx : np.where(regressor(tx) < threshold, 0, 1)


##### GENERIC IMPLEMENTATION OF TRAINING FUNCTION


"""
Learns the model by supplying the tuple "learn_args" to the learning function "learn".
Returns the model (w, loss) and the corresponding regressor and classifier.
"""
def train(learn_args, learn, get_regressor, threshold):
    # Learn the model
    w, loss = learn(*learn_args)
    
    # Get the regressor
    regressor = get_regressor(w)
    # Get the classifier
    classifier = get_classifier(regressor, threshold)
    
    return w, loss, regressor, classifier


##### ACTUAL TRAINING FUNCTIONS: MSE


""" Training for L2-regularised MSE with Gradient Descent """
def train_reg_mse_GD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma)
    return train(learn_args, reg_least_squares_GD, get_linear_regressor, threshold)


""" Training for L2-regularised MSE with Stochastic Gradient Descent """
def train_reg_mse_SGD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma)
    return train(learn_args, reg_least_squares_SGD, get_linear_regressor, threshold)


""" Training for L2-regularised MSE with Normal Equations """
def train_reg_mse_NE(y, tx, lambda_, threshold):
    learn_args = (y, tx, lambda_)
    return train(learn_args, ridge_regression, get_linear_regressor, threshold)


""" Training for unregularised MSE with Gradient Descent """
def train_unreg_mse_GD(y, tx, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, initial_w, max_iters, gamma)
    return train(learn_args, least_squares_GD, get_linear_regressor, threshold)


""" Training for unregularised MSE with Stochastic Gradient Descent """
def train_unreg_mse_SGD(y, tx, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, initial_w, max_iters, gamma)
    return train(learn_args, least_squares_SGD, get_linear_regressor, threshold)


""" Training for unregularised MSE with Normal Equations """
def train_unreg_mse_NE(y, tx, threshold):
    learn_args = (y, tx)
    return train(learn_args, least_squares, get_linear_regressor, threshold)


##### ACTUAL TRAINING FUNCTIONS: LOGISTIC


""" Training for L2-regularised logistic loss with Gradient Descent """
def train_reg_log_GD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma)
    return train(learn_args, reg_logistic_regression, get_linear_regressor, threshold)


""" Training for L2-regularised logistic loss with Stochastic Gradient Descent """
def train_reg_log_SGD(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, lambda_, initial_w, max_iters, gamma)
    return train(learn_args, reg_logistic_regression_SGD, get_linear_regressor, threshold)


""" Training for unregularised logistic loss with Gradient Descent """
def train_unreg_log_GD(y, tx, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, initial_w, max_iters, gamma)
    return train(learn_args, logistic_regression, get_linear_regressor, threshold)


""" Training for unregularised logistic loss with Stochastic Gradient Descent """
def train_unreg_log_SGD(y, tx, initial_w, max_iters, gamma, threshold):
    learn_args = (y, tx, initial_w, max_iters, gamma)
    return train(learn_args, logistic_regression_SGD, get_linear_regressor, threshold)