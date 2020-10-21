"""
This file contains functions useful to manipulate data.
"""

# -*- coding: utf-8 -*-
import numpy as np


def standardise(x):
    """" Standardise the original data set. """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def split_data(y, tx, train_ratio):
    """ Splits the data between training and testing. """
    N = len(y)
    indices = np.random.permutation(N)
    
    tx = tx[indices, :]
    y = y[indices]
    
    bound = int(N * train_ratio)
    return y[:bound], tx[:bound, :], y[bound:], tx[bound:, :]