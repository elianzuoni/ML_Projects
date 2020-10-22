"""
The file containing the implementations of the 6 required basic ML functions + their dependencies + other similar functions
"""

# -*- coding: utf-8 -*-
import numpy as np


##### UTILITY COMPUTATION FUNCTIONS


def compute_reg_ls_loss(y, tx, w, lambda_=0):
    """ Computes the value of the L2-regularised Least-Squares loss function, evaluated at point w. """
    N = len(y)
    e = y - np.dot(tx, w)
    
    loss = (0.5/N) * np.dot(e.T, e)
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    return loss


def compute_reg_ls_gradient(y, tx, w, lambda_=0):
    """ Computes the gradient of the L2-regularised Least-Squares loss function, evaluated at point w. """
    e = y - np.dot(tx, w)
    grad = -(1/len(y)) * np.dot(tx.T, e)
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    return grad


def sigmoid(t):
    """ Compute the sigmoid function, evaluated at point t. """
    return np.exp(t) / (1+np.exp(t))


def compute_reg_logistic_loss(y, tx, w, lambda_):
    """ Compute the value of the L2-regularised logistic loss function (negative log-likelihood). """
    # Vectorised computation
    z = np.dot(tx, w)
    loss = np.sum(np.log(1 + np.exp(z))) - np.dot(y.T, z)
    
    # Regularise, if necessary
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    #print("LogLoss. lambda_ =", lambda_, ", loss =", loss, ", w =", w)
    return loss


def compute_reg_logistic_gradient(y, tx, w, lambda_):
    """ Compute the gradient of the L2-regularised logistic loss function, evaluated at point w. """
    diff = sigmoid(np.dot(tx, w)) - y
    
    grad = np.dot(tx.T, diff)
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    #print("LogGrad. lambda_ =", lambda_, ", grad =", grad, ", w =", w)
    return grad


##### GENERIC IMPLEMENTATIONS OF (REGULARISED) GD AND SGD


def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_loss, compute_gradient):
    """ Implements the Gradient Descent algorithm for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """
    # These values will be updated iteratively
    w = initial_w
    loss = compute_loss(y, tx, w, lambda_)
    
    # Always do max_iters iterations, no matter what
    for n_iter in range(max_iters):
        # Compute the gradient evaluated at the current point w
        grad = compute_gradient(y, tx, w, lambda_)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, lambda_)

    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, batch_size, compute_loss, compute_gradient):
    """ Implements the Stochastic Gradient Descent algorithm for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """    
    # These values will be updated iteratively
    w = initial_w
    loss = compute_loss(y, tx, w, lambda_)
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
        grad = compute_gradient(y_sub, tx_sub, w, lambda_)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, lambda_)
        
        # Move to next batch
        n_batch += 1
        
    return w, loss


##### REQUIRED ML FUNCTIONS


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient Descent for the Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_ls_loss, compute_reg_ls_gradient)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """ Stochastic Gradient Descent for the Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, batch_size, compute_reg_ls_loss, compute_reg_ls_gradient)


def least_squares(y, tx):
    """ Computes the exact solution to the Least-Squares minimisation problem using normal equations. """
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_reg_ls_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Computes the exact solution to the L2-regularised Least-Squares minimisation problem using normal equations. """
    lambda_prime = lambda_ * 2 * len(y)
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_prime*np.eye(tx.shape[1]), np.dot(tx.T, y))
    loss = compute_reg_ls_loss(y, tx, w, lambda_)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Gradient Descent for the logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_logistic_loss, compute_reg_logistic_gradient)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Gradient Descent for the L2-regularised logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_logistic_loss, compute_reg_logistic_gradient)


##### NON-REQUIRED ML FUNCTIONS


def reg_least_squares_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Gradient Descent for the L2-regularised Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_ls_loss, compute_reg_ls_gradient)


def reg_least_squares_SGD(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1):
    """ Stochastic Gradient Descent (with batch-size 1) for the L2-regularised Least-Squares cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, batch_size, compute_reg_ls_loss, compute_reg_ls_gradient)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """ Stochastic Gradient Descent for the logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, batch_size, compute_reg_logistic_loss, compute_reg_logistic_gradient)


def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1):
    """ Stochastic Gradient Descent for the L2-regularised logistic cost function. """
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, batch_size, compute_reg_logistic_loss, compute_reg_logistic_gradient)