"""
This file contains functions useful to manipulate data.
"""

# -*- coding: utf-8 -*-
import numpy as np


""" Splits the data between training and testing. """
def split_data(y, tx, train_ratio):
    N = len(y)
    indices = np.random.permutation(N)
    
    tx = tx[indices, :]
    y = y[indices]
    
    bound = int(N * train_ratio)
    return y[:bound], tx[:bound, :], y[bound:], tx[bound:, :]