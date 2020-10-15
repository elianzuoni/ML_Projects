"""
The file containing the implementations of the 6 required basic ML functions.
"""

# -*- coding: utf-8 -*-
import numpy as np


##### UTILITY FUNCTIONS


""" Computes the value of the MSE loss function, evaluated at point w. """
def compute_mse_loss(y, tx, w):
    N = len(y)
    e = y - np.dot(tx, w)
    return (0.5/N) * np.linalg.norm(e, ord=2)**2


""" Computes the gradient of the MSE loss function, evaluated at point w. """
def compute_mse_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    grad = -(1/len(y)) * np.dot(tx.T, e)
    return grad


##### REQUIRED FUNCTIONS


""" Implements the Gradient Descent algorithm for the MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # These values will be updated iteratively
    w = initial_w
    loss = compute_mse_loss(y, tx, w)
    
    # Always do max_iters iterations, no matter what
    for n_iter in range(max_iters):
        # Compute the gradient evaluated at the current point w
        grad = compute_mse_gradient(y, tx, w)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_mse_loss(y, tx, w)

    return w, loss


""" Implements the Stochastic Gradient Descent algorithm (with batch-size 1) for the MSE cost function.
    Returns the last weight vector and the corresponding loss. """
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # These values will be updated iteratively
    w = initial_w
    loss = compute_mse_loss(y, tx, w)
    
    np.random.seed()
    for n_iter in range(max_iters):
        # Sample index of datapoint
        n = np.random.randint(0, len(y))
        
        # Subsample y and tx
        y_sub = y[n]
        tx_sub = tx[n, :]
        
        # Compute the (stochastic approximation of the) gradient evaluated at the current point w
        grad = compute_gradient[method](y_sub, tx_sub, w)
        
        # Update w, and the corresponding loss
        w = w - gamma * grad
        loss = compute_mse_loss(y, tx, w)
    
    return w, loss


""" Computes the exact solution to the MSE minimisation problem using normal equations. """
def least_squares(y, tx):
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss


""" Computes the exact solution to the L2-regularised MSE minimisation problem using normal equations. """
def ridge_regression(y, tx, lambda_):
    # Closed formula for optimal w
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_*np.eye(tx.shape[1]), np.dot(tx.T, y))
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss