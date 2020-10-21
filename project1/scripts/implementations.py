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


"""apply the sigmoid function on t"""
def sigmoid(t):
    return (np.exp(t)/(1+np.exp(t)))


"""compute the loss for logistic regression: negative log likelihood. """

def calculate_loss_logistic(y, tx, w):
    loss=0 #initialisation of the loss
    for i in range (len(y)) :
        XiT_W = np.transpose(tx[i,:]).dot(w) #calculation of (xi)T * w
        loss+=np.log(1+np.exp(XiT_W)) - y[i]*XiT_W
    return (loss)


"""compute the gradient of loss."""
def calculate_gradient_logistic(y, tx, w):
    Xt = np.transpose(tx) #X transpose
    diff = sigmoid(tx.dot(w)) - y #sigmoid(X*w)-y
    return (np.dot(Xt, diff))


""" Do one step of gradient descent using logistic regression.
Regularization taken into account, set to 0 if no regularization coefficient lambda_ specified
Return the loss and the updated w."""
def logistic_step_GD(y, tx, w, gamma, lambda_= 0.):
    loss = calculate_loss_logistic(y,tx,w)
    gradient = calculate_gradient_logistic(y,tx,w)
    
    #we update the loss and gradient in case there is regularization
    loss = loss + lambda_*np.linalg.norm(w) #penalized loss
    gradient = gradient + lambda_*2*w #we add the gradient of the norm
    w = w - gamma*gradient
    return loss, w

""" Do one step of stochastic gradient descent using logistic regression. 
Regularization taken into account, set to 0 if no regularization coefficient lambda_ specified
Return the loss and the updated w."""
def logistic_step_SGD(y, tx, w, gamma, lambda_= 0.):
    np.random.seed()
    rand_nb = np.random.randint(0, len(y)) #select a random number between 0 and N-1 : to select a random sample
    
    #same as GD but with just one random sample
    loss = calculate_loss_logistic(y[rand_nb],tx[rand_nb,:],w)
    gradient = calculate_gradient_logistic(y[rand_nb],tx[rand_nb,:],w)
    
    #we update the loss and gradient in case there is regularization
    loss = loss + lambda_*np.linalg.norm(w) #penalized loss
    gradient = gradient + lambda_*2*w #we add the gradient of the norm
    w = w - gamma*gradient
    return loss, w



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
        grad = compute_mse_gradient(y_sub, tx_sub, w)
        
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



''' Logistic regression using gradient descent or SGD : here we use GD (uncomment SGD to apply SGD)'''
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    w = initial_w
    losses=[]
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = logistic_step_GD(y, tx, w, gamma) #GD
        #loss, w = logistic_step_SGD(y, tx, w, gamma) #SGD
        
        # converge criterion : if the loss don't change anymore, stop the algo
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold: 
            break
            
    return w, loss


def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    threshold = 1e-8
    w = initial_w
    losses=[]
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = logistic_step_GD(y, tx, w, gamma, lambda_) #GD
        #loss, w = logistic_step_SGD(y, tx, w, gamma, lambda_) #SGD
        
        # converge criterion : if the loss don't change anymore, stop the algo
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold: 
            break
            
    return w, loss

    