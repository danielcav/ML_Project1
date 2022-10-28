import numpy as np
import seaborn as sns
from implementations import *

# Data loading
y, tx, ids = load_csv_data("C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")
x_control = tx

# Data processing
x_control = (min_max_scaling(x_control))
x_test, to_replace = data_processing(y, tx)
# Initializing weights
w_control = np.ones(x_control.shape[1])
w_test = np.ones(x_test.shape[1])

w2, loss2 = ridge_regression(y, x_test, 0.0001)
"""
# Put back missing features for the submission
for replace in to_replace:
    w2 = np.insert(w2, replace, 0)
"""
##################################
"""
# Run ML algorithms with little data processing (control) and processed data (test)
w1, loss_control, = ridge_regression(y, x_control, 0.00001)
w2, loss2 = ridge_regression(y, x_test, 0.00001)
print(loss_control, loss2)
"""

"""
# PCA :
mean = tx.mean(axis=0)
C = tx - mean
V = np.cov(C, rowvar=False)
values, vectors = np.linalg.eig(V)
print(vectors.shape)
T = tx.dot(vectors[:, :10])
print(T.shape)
#explain_ratio = np.sum(values[:10])/np.sum(values)
#print(explain_ratio)
"""
