from implementations import *


y, tx, ids = load_csv_data("C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")

for num, row in enumerate(tx.T):
    tx.T[num] = reduce_outliers(row)

w = np.ones(30)


#print(x_clean)
#x_clean = (np.abs(tx) > mean_tx + 2 * std_tx)
# print(x_clean)
w, loss = mean_squared_error_gd(y, tx, w, 100, 0.1)
print(loss)


