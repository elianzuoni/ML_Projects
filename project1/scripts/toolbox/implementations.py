"""
This file contains the implementations of the 6 required basic ML functions, their dependencies (functions to compute losses and gradients), and other similar non-required ML functions (e.g. regularised Least-Squares with SGD).  
The sigmoid function and the Logistic loss function needed particular care to be implemented in a numerically stable way, i.e. to avoid cancellation of large numbers (often resulting in computations like inf/inf and inf-inf, respectively). The Logistic loss and gradient functions accept a parameter specifying whether the loss/gradient should be normalised by the number of datapoints (as is the case for Least-Squares), so as to make SGD correctly estimate the magnitude of the gradient.  
GD and SGD are implemented in a generic way: concrete ML functions simply call these generic implementations with the right parameters and loss/gradient functions.
"""

# -*- coding: utf-8 -*-
import numpy as np


##### UTILITY COMPUTATION FUNCTIONS


def compute_reg_ls_loss(y, tx, w, lambda_, normalise=False):
    """ Computes the value of the L2-regularised Least-Squares loss function, evaluated at point w.
    The parameter "normalise" is unused. """
    N = len(y)
    e = y - np.dot(tx, w)
    
    loss = (0.5/N) * np.dot(e.T, e)
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    return loss


def compute_reg_ls_gradient(y, tx, w, lambda_, normalise=False):
    """ Computes the gradient of the L2-regularised Least-Squares loss function, evaluated at point w.
    The parameter "normalise" is unused. """
    e = y - np.dot(tx, w)
    grad = -(1/len(y)) * np.dot(tx.T, e)
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    return grad


def sigmoid(t):
    """ Compute the sigmoid function, evaluated at point t. """
    # Numerical stabilisation
    return np.piecewise(t, [t > 0], [lambda x : 1/(1 + np.exp(-x)), lambda x : np.exp(x)/(1 + np.exp(x))])


def compute_reg_logistic_loss(y, tx, w, lambda_, normalise):
    """ Compute the value of the L2-regularised logistic loss function (negative log-likelihood). """
    # Vectorised computation
    z = np.dot(tx, w)
    # Numerically stable implementation
    loss =  np.sum(np.log( np.exp(-y*z) + np.exp((1-y)*z) ))
    
    # Normalise by number of datapoints, to make SGD work
    if normalise:
        loss /= tx.shape[0]
    
    # Regularise, if necessary
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    return loss


def compute_reg_logistic_gradient(y, tx, w, lambda_, normalise):
    """ Compute the gradient of the L2-regularised logistic loss function, evaluated at point w. """
    diff = sigmoid(np.dot(tx, w)) - y
    
    grad = np.dot(tx.T, diff)
    # Normalise by number of datapoints, to make SGD work
    if normalise:
        grad /= tx.shape[0]
    
    # Regularise, if necessary
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    return grad


##### GENERIC IMPLEMENTATIONS OF (REGULARISED) GD AND SGD


def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, normalise, compute_loss, compute_gradient):
    """ Implements the Gradient Descent algorithm for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """
    # These values will be updated iteratively
    w = initial_w
    loss = compute_loss(y, tx, w, lambda_, normalise)
    
    # Always do max_iters iterations, no matter what
    for n_iter in range(max_iters):
        # Compute the gradient evaluated at the current point w
        grad = compute_gradient(y, tx, w, lambda_, normalise)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, lambda_, normalise)

    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, normalise, batch_size, compute_loss, compute_gradient):
    """ Implements the Stochastic Gradient Descent algorithm for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """    
    # These values will be updated iteratively
    w = initial_w
    loss = compute_loss(y, tx, w, lambda_, normalise)
    N = len(y)
    
    if batch_size > N:
        raise Exception("batch_size > N")
    
    # Permuted indices
    indices = np.random.permutation(N)
    # Current batch
    n_batch = 0
    for n_iter in range(max_iters):
        # New epoch
        if (n_batch+1)*batch_size > N:
            indices = np.random.permutation(N)
            n_batch = 0
        
        # Sample indices of datapoints
        sub = indices[n_batch*batch_size : (n_batch+1)*batch_size]
        
        # Subsample y and tx
        y_sub = y[sub]
        tx_sub = tx[sub, :]
        
        # Compute the (stochastic approximation of the) gradient evaluated at the current point w
        grad = compute_gradient(y_sub, tx_sub, w, lambda_, normalise)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, lambda_, normalise)
        
        # Move to next batch
        n_batch += 1
        
    return w, loss


##### REQUIRED ML FUNCTIONS


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient Descent for the Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, False, compute_reg_ls_loss, compute_reg_ls_gradient)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1, normalise=False):
    """ Stochastic Gradient Descent for the Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, normalise, batch_size, compute_reg_ls_loss, compute_reg_ls_gradient)


def least_squares(y, tx):
    """ Computes the exact solution to the Least-Squares minimisation problem using normal equations. """
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_reg_ls_loss(y, tx, w, 0)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Computes the exact solution to the L2-regularised Least-Squares minimisation problem using normal equations. """
    lambda_prime = lambda_ * 2 * len(y)
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_prime*np.eye(tx.shape[1]), np.dot(tx.T, y))
    loss = compute_reg_ls_loss(y, tx, w, lambda_)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, normalise=False):
    """ Gradient Descent for the logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, normalise, compute_reg_logistic_loss, compute_reg_logistic_gradient)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, normalise=False):
    """ Gradient Descent for the L2-regularised logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, normalise, compute_reg_logistic_loss, compute_reg_logistic_gradient)


##### NON-REQUIRED ML FUNCTIONS


def reg_least_squares_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Gradient Descent for the L2-regularised Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, False, compute_reg_ls_loss, compute_reg_ls_gradient)


def reg_least_squares_SGD(y, tx, lambda_, initial_w, max_iters, gamma, batch_size):
    """ Stochastic Gradient Descent (with batch-size 1) for the L2-regularised Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, False, batch_size, compute_reg_ls_loss, compute_reg_ls_gradient)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size, normalise):
    """ Stochastic Gradient Descent for the logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, normalise, batch_size, compute_reg_logistic_loss, compute_reg_logistic_gradient)


def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma, batch_size, normalise):
    """ Stochastic Gradient Descent for the L2-regularised logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, normalise, batch_size, compute_reg_logistic_loss, compute_reg_logistic_gradient)