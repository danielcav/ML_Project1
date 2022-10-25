from implementations import *

# Data loading
y, tx, ids = load_csv_data("C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")

# Data processing
x = tx.T
col = []
for index, feature in enumerate(x):
    percent = np.count_nonzero(feature == -999)/2500
    if percent >= 70:
        col.append(index)
x = np.delete(x, col, axis=0)
x_clean = x.T

for num, row in enumerate(tx.T):
    tx.T[num] = reduce_outliers(row)

for num, row in enumerate(x_clean.T):
    x_clean.T[num] = reduce_outliers(row)

# Initialisation of weights
w_clean = np.zeros(x_clean.shape[1])
w = np.zeros(tx.shape[1])

# Running algorithms
#w_control, loss_control = reg_logistic_regression(y, tx, 0.1, w, 1000, 0.03)
w_test, loss_test = logistic_regression(y, x_clean, w_clean, 1500, 0.03)
print( "\n", loss_test)


#for num, row in enumerate(tx.T):
#    tx.T[num] = reduce_outliers(row)

#w = np.ones(tx.shape[1])
#w, loss = logistic_regression(y, tx, w, 10, 0.01)


#print(x_clean)
#x_clean = (np.abs(tx) > mean_tx + 2 * std_tx)
# print(x_clean)



