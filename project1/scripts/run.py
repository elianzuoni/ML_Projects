""" This file contains the entire workflow, from data loading to prediction generation. The training is done with the best (hyper)parameters found with model selection in project1.ipynb. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from toolbox.manipulate_data import *
from toolbox.training import *

# Set seed to ensure reproducibility
seed = 1
np.random.seed(seed)


# Fixed parameters, not to be changed
null = -999 # The placeholder for missing values
drop_thresholds = [0.1, 0.33, 0.7, 1] # There are only 4 levels of feature missingness
class_thresh = 0.5 # The threshold for the regressor-based classifier

# Best training parameters found with model selection in the notebook
# Do not delete them to try different ones: comment them out (if new submission is worse, we still need the best parameters here)
# drop_thresh = drop_thresholds[2]
drop_thresh = drop_thresholds[2]
# degree = 5
degree = 10
# max_iters = 10000
# batch_size = 2000
# gamma = 1e-5
# lambda_ = 1e-4
lambda_ = 1e-3
# initial_w = np.zeros(tX.shape[1])


# Load raw training dataset
DATA_TRAIN_PATH = '../data/train.csv'
y_raw, X_raw, ids = load_csv_data(DATA_TRAIN_PATH)

# Clean dataset
X = clean_data(X_raw, null, drop_thresh)

# Standardise dataset
X, mean_X, std_X = standardise(X)

# Map y from bilateral {-1, +1} domain to unilateral {0, 1} domain
y = bin_bil_to_unil(y_raw)

# Expand features
tX = expand_features(X, degree)


# Train using the best method found with model selection in the notebook
# As for parameters, comment this line out to try a different one
w, train_loss, regressor, classifier = train_reg_ls_NE(y, tX, lambda_, class_thresh)
# w, train_loss, regressor, classifier = train_reg_ls_NE(y, tX, lambda_, class_thresh)


# Load challenge data
DATA_TEST_PATH = '../data/test.csv'
_, X_ch_raw, ids_ch = load_csv_data(DATA_TEST_PATH)

# Clean dataset
X_ch = clean_data(X_ch_raw, null, drop_thresh)

# Standardise dataset
X_ch, mean_X_ch, std_X_ch = standardise(X_ch)

# Expand features
tX_ch = expand_features(X_ch, degree)


# Get prediction in {0, 1}
y_unil = classifier(tX_ch)

# Map it to {-1, +1}
y_bil = bin_unil_to_bil(y_unil)

# Write formatted output file
OUTPUT_PATH = 'submission.csv'
create_csv_submission(ids_ch, y_bil, OUTPUT_PATH)
print("Output file created")