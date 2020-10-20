"""
The file containing the implementations of the 6 required basic ML functions + their dependencies + other similar functions
"""

# -*- coding: utf-8 -*-
import numpy as np


##### UTILITY COMPUTATION FUNCTIONS


""" Computes the value of the L2-regularised MSE loss function, evaluated at point w. """
def compute_reg_mse_loss(y, tx, w, lambda_=0):
    N = len(y)
    e = y - np.dot(tx, w)
    
    loss = (0.5/N) * np.dot(e.T, e)
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    return loss


""" Computes the gradient of the L2-regularised MSE loss function, evaluated at point w. """
def compute_reg_mse_gradient(y, tx, w, lambda_=0):
    e = y - np.dot(tx, w)
    grad = -(1/len(y)) * np.dot(tx.T, e)
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    return grad


""" Compute the sigmoid function, evaluated at point t. """
def sigmoid(t):
    return (np.exp(t)/(1+np.exp(t)))


""" Compute the value of the L2-regularised logistic loss function (negative log-likelihood). """
def compute_reg_logistic_loss(y, tx, w, lambda_=0):
    ## TODO: if possible, compact it into vectorised operation
    loss=0
    for i in range (len(y)) :
        XiT_W = np.transpose(tx[i,:]).dot(w) # Calculation of (xi)T * w
        loss+=np.log(1+np.exp(XiT_W)) - y[i]*XiT_W
        
    # Regularise, if necessary
    if lambda_ > 0:
        loss += lambda_ * np.dot(w.T, w)
    return loss


""" Compute the gradient of the L2-regularised logistic loss function, evaluated at point w. """
def compute_reg_logistic_gradient(y, tx, w, lambda_=0):
    diff = sigmoid(tx.dot(w)) - y
    
    grad = np.dot(tx.T, diff)
    if lambda_ > 0:
        grad += lambda_ * 2 * w
    return grad


##### GENERIC IMPLEMENTATIONS OF (REGULARISED) GD AND SGD


""" Implements the Gradient Descent algorithm for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """
def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_loss, compute_gradient):
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


""" Implements the Stochastic Gradient Descent algorithm (with batch-size 1) for the provided loss and gradient functions.
    Returns the last weight vector and the corresponding loss. """
def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_loss, compute_gradient):
    # These values will be updated iteratively
    w = initial_w
    loss = compute_loss(y, tx, w, lambda_)
    N = len(y)
    
    # Will contain permuted indices
    indices = []
    for n_iter in range(max_iters):
        # New epoch each N iterations
        if n_iter % N == 0:
            indices = np.random.permutation(N)
        
        # Sample index of datapoint
        n = indices[n_iter % N]
        
        # Subsample y and tx
        y_sub = y[n:n+1]   # This way it is still a vector
        tx_sub = tx[n:n+1, :]   # This way it is still a matrix
        
        # Compute the (stochastic approximation of the) gradient evaluated at the current point w
        grad = compute_gradient(y_sub, tx_sub, w, lambda_)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, lambda_)
        
    return w, loss


##### REQUIRED ML FUNCTIONS


""" Implements the Gradient Descent algorithm for the MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_mse_loss, compute_reg_mse_gradient)


""" Implements the Stochastic Gradient Descent algorithm (with batch-size 1) for the MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_mse_loss, compute_reg_mse_gradient)


""" Computes the exact solution to the MSE minimisation problem using normal equations. """
def least_squares(y, tx):
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_reg_mse_loss(y, tx, w)
    
    return w, loss


""" Computes the exact solution to the L2-regularised MSE minimisation problem using normal equations. """
def ridge_regression(y, tx, lambda_):
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_*np.eye(tx.shape[1]), np.dot(tx.T, y))
    loss = compute_reg_mse_loss(y, tx, w, 0)
    
    return w, loss


""" Implements the Gradient Descent algorithm for the logistic cost function.
    Returns the last weight vector and the corresponding loss. """
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_logistic_loss, compute_reg_logistic_gradient)


""" Implements the Gradient Descent algorithm for the L2-regularised logistic cost function.
    Returns the last weight vector and the corresponding loss. """
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_logistic_loss, compute_reg_logistic_gradient)


##### NON-REQUIRED ML FUNCTIONS


""" Implements the Gradient Descent algorithm for the L2-regularised MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def reg_least_squares_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_mse_loss, compute_reg_mse_gradient)


""" Implements the Stochastic Gradient Descent algorithm (with batch-size 1) for the L2-regularised MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def reg_least_squares_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_mse_loss, compute_reg_mse_gradient)


""" Implements the Stochastic Gradient Descent algorithm for the logistic cost function.
    Returns the last weight vector and the corresponding loss. """
def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 0, compute_reg_logistic_loss, compute_reg_logistic_gradient)


""" Implements the Stochastic Gradient Descent algorithm for the L2-regularised logistic cost function.
    Returns the last weight vector and the corresponding loss. """
def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    # Just provide the right loss and gradient functions to the generic implementation
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_, compute_reg_logistic_loss, compute_reg_logistic_gradient)