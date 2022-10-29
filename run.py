import numpy as np
import seaborn as sns
from implementations import *

# Data loading
y, tx, ids = load_csv_data("C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")
# Data processing
x_test, to_replace = data_processing(y, tx)
features = np.arange(0, x_test.shape[1], 1)
degree = np.arange(2, 21, 1)
_, loss = least_squares(y, x_test)
losses = np.array([0, loss])

for deg in degree:
    x_test_aug = polynomial_features(x_test, features, deg)
    _, loss2 = least_squares(y, x_test_aug)
    loss_deg = np.array([deg, loss2])
    losses = np.column_stack((losses, loss_deg))

plt.plot(losses[0], losses[1])
plt.show()

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
