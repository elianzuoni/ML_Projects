"""
This file contains functions useful to manipulate data.
"""

# -*- coding: utf-8 -*-
import numpy as np



"""" Standardise the original data set. """
def standardise(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


""" Splits the data between training and testing. """
def split_data(y, tx, train_ratio):
    N = len(y)
    indices = np.random.permutation(N)
    
    tx = tx[indices, :]
    y = y[indices]
    
    bound = int(N * train_ratio)
    return y[:bound], tx[:bound, :], y[bound:], tx[bound:, :]

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N=x.shape[0]
    D=x.shape[1]
    PHI = np.ones((N,D))
    for deg in range (1,degree+1) :
        PHI = np.c_[PHI, np.power(x, deg)] 
    return PHI


def clean_data(x):
    """function that removes unnecessary features,
    and replace the '-999.0' values by 0"""
    #x = np.delete(x, np.s_[0:13], axis=1) 
    x[x==-999.0] = 0
    return x