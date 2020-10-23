"""
This file contains functions useful to manipulate data.
"""

# -*- coding: utf-8 -*-
import numpy as np


def clean_data(x, null, drop_thresh):
    """ Returns a cleaned version of the data, i.e. with no "null" entries.
    It does so, first, by dropping features for which at least a fraction "drop_thresh" of the datapoints have a null entry.
    After this first phase, the remaining null values are substituted with the mean (over non-null entries) of that feature. """
    # Do not modify the original dataset
    x_clean = np.copy(x)
    
    # Vector holding, for each feature, the fraction of datapoints with a null value
    null_frac = (1/x_clean.shape[0]) * np.sum(x_clean==null, axis=0)
    # Boolean vector holding, for each feature, whether or not it needs to be kept
    column_to_keep = null_frac <= drop_thresh
    
    # Drop bad columns
    x_clean = x_clean[:, column_to_keep]
    
    # Vector of (list of) indices of columns where there are still null values
    columns_to_interpolate = np.argwhere(np.any(x_clean==null, axis=0))
    
    # For each of those columns, find the mean of non-null values, and substitute it to null values
    for col_list in columns_to_interpolate:
        # Extrapolate only entry of col_list
        col = col_list[0]
        
        # Boolean vector holding, for each row, whether or not it has a "null" entry at position "col"
        row_non_null = x_clean[:, col] != null
        # Find mean
        interpolation = np.mean(x_clean[row_non_null, col])
        # Substitute it to null values
        row_null = np.logical_not(row_non_null)
        x_clean[row_null, col] = interpolation
    
    return x_clean


def standardise(x):
    """" Standardise the original data set. """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def bin_unil_to_bil(a):
    """ Maps (vectors in) {0, 1} to (vectors in) {-1, 1} """
    return 2*a - 1


def bin_bil_to_unil(a):
    """ Maps (vectors in) {-1, 1} to (vectors in) {0, 1} """
    return (a + 1)/2


def expand_features(x, degree):
    """ Polynomial expansion of features (all features with the same degree) """
    N = x.shape[0]
    D = x.shape[1]
    
    # Matrix to be returned
    phi = np.ones((N, 1))
    # Holds X^deg
    xdeg = np.ones((N, D))
    for deg in range (1,degree+1) :
        xdeg *= x
        phi = np.c_[phi, xdeg] 
    
    return phi


def split_data(y, tx, train_ratio):
    """ Splits the data between training and testing. """
    N = len(y)
    indices = np.random.permutation(N)
    
    tx = tx[indices, :]
    y = y[indices]
    
    bound = int(N * train_ratio)
    return y[:bound], tx[:bound, :], y[bound:], tx[bound:, :]