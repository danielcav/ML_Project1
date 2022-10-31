import numpy as np
import seaborn as sns
from implementations import *
from helpers import *

######## UNZIP DATA BEFORE RUNNING #############

# download train data
SET_TRAIN = 'data/train.csv' 
y, tx, ids = load_csv_data(SET_TRAIN)

# Initialization
degree = np.arange(2, 25, 1)
lambdas = np.logspace(-9, -6, 5)

#Choosing best parameters based on previous observations explained in report
best_lambda = lambdas[0]
best_degree = degree[19]

# Training
weights, losses = train(y, tx)

# download test data
SET_TEST = 'data/test.csv' 
_, tx_test, ids_test = load_csv_data(SET_TEST)

# Testing
Y_pred = test(tx_test, best_lambda, best_degree, weights)

# Submission
OUTPUT= 'Final_submission.csv' 
final_pred = predict_labels_split(Y_pred)
create_csv_submission(ids_test, final_pred, OUTPUT)