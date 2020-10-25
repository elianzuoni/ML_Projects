# Project 1

The code is contained in the [scripts](scripts) folder.  
The subfolder [toolbox](scripts/toolbox) contains files implementing the building blocks of a complete ML pipeline.  
The pipeline, including model selection, is implemented in the notebook [project1.ipynb](scripts/project1.ipynb), located at the top level; the script [run.py](scripts/run.py) contains the same pipeline, without model selection (the best parameters are copied from the notebook).  

Following is the description of each file.


## Toolbox

### [implementations.py](scripts/toolbox/implementations.py) 
This file contains the implementations of the 6 required basic ML functions, their dependencies (functions to compute losses and gradients), and other similar non-required ML functions (e.g. regularised Least-Squares with SGD).  
The sigmoid function and the Logistic loss function needed particular care to be implemented in a numerically stable way, i.e. to avoid cancellation of large numbers (often resulting in computations like inf/inf and inf-inf, respectively). The Logistic loss and gradient functions accept a parameter specifying whether the loss/gradient should be normalised by the number of datapoints (as is the case for Least-Squares), so as to make SGD correctly estimate the magnitude of the gradient.  
GD and SGD are implemented in a generic way: concrete ML functions simply call these generic implementations with the right parameters and loss/gradient functions.

### [training.py](scripts/toolbox/training.py) 
This file contains training functions, adapting those in implementations.py, offering an intuitive and homogeneous interface.  
The functions are named "train_(reg/unreg)_<cost>_<method>": the name specifies the cost function ("ls" for Least-Squares, "log" for Logistic), whether or not it is regularised, and the method used to optimise it (GD, SGD, or NE, for Normal Equations).  
These functions return (w, train_loss, regressor, classifier), where w is the optimal weight vector, train_loss is the loss on the training dataset, regressor is a function (parametrised by w) mapping datapoints to continuous predictions, and classifier is the function (parametrised by the regressor and a threshold) mapping datapoints to predictions in {0, 1}.

### [manipulate_data.py](scripts/toolbox/manipulate_data.py) 
This file contains utility functions that manipulate the dataset.  
2 functions act on the labels, to map them from {-1, +1} to {0, 1}, and viceversa.  
4 functions act on the datapoints: one to clean it (i.e. rid it of null values), one to standardise it, one to expand its features, and one to split it, according to a given ratio, between a training and a testing dataset.

### [testing.py](scripts/toolbox/testing.py) 
This file contains functions useful for testing, either "manually" (i.e. by feeding a manually-trained predictor and a manually-extracted testing dataset to an assesser), or "automatically" (i.e. via cross validation, which does the splitting, the training, and the testing).  
A different metric is used for the test loss than for the train loss (i.e. the one minimised by the training): for example, when the classifier is being assessed,it is the fraction of misclassified points, which more closely reflects the actual goodness of fit. 

### [toolbox.ipynb](scripts/toolbox/toolbox.ipynb) 
This notebook contains tests for the functions implemented in this folder, run on randomly-generated datasets.

## Top level

### [proj1_helpers.py](scripts/proj1_helpers.py) 
Helpers for I/O with csv files, as given by the instructors.

### [project1.ipynb](scripts/project1.ipynb) 
This notebook contains the entire workflow, from data loading and preprocessing, through model selection and training with tuned parameters, to label prediction on the challenge dataset.

### [run.py](scripts/run.py) 
This file is almost a replica of the content of [project1.ipynb](scripts/project1.ipynb), except for model selection: the best parameters are directly copied over from the notebook, without recomputing them.