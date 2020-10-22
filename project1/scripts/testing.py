"""
This file contains functions useful for testing.
"""

# -*- coding: utf-8 -*-
import numpy as np


##### FUNCTIONS COMPUTING TEST LOSS


""" Returns the mean square difference between a and b (continuous values). """
def compute_mse(a, b):
    return (1 / len(a)) * np.dot((a-b).T, a-b)


""" Returns the normalised Hamming distance between a and b (binary values). """
def compute_nhd(a, b):
    return (1 / len(a)) * np.sum(np.where(a != b, 1, 0))


##### GENERIC IMPLEMENTATION OF PREDICTOR ASSESSER


""" Returns the loss (specified by "compute_test_loss") between y_test and the prediction on tx_test """
def assess_predictor(y_test, tx_test, predictor, compute_test_loss):
    return compute_test_loss(y_test, predictor(tx_test))


##### ACTUAL PREDICTOR ASSESSERS


""" Assesses the provided regressor with MSE loss function """
def assess_regressor_mse(y_test, tx_test, regressor):
    return assess_predictor(y_test, tx_test, regressor, compute_mse)


""" Assesses the provided classifier with NHD loss function """
def assess_classifier_nhd(y_test, tx_test, classifier):
    return assess_predictor(y_test, tx_test, classifier, compute_nhd)

