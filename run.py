from implementations import *


#y, tx, ids = load_csv_data("C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")
# y, mean_y, std_y = standardize(y)
# tx, mean_tx, std_tx = standardize(tx)
#w = np.ones(30)
# w, losses = logistic_regression(y, tx, w, 10, 0.1)
#x = tx[:, 5]
#x = x[x != -999.0]
#print(len(x))



# print(y.max(), np.sort(tx.max(axis=0))[-6:-1])
# initial_w = np.ones(30)
#initial_w = np.array([[0.409111], [0.843996]])
# initial_w = np.array([[0.463156], [0.939874]])
initial_w = np.array([[0.5], [1.0]])
y = np.array([[0.1], [0.3], [0.5]])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
y = (y > 0.2) * 1.0
w, loss = reg_logistic_regression(y, tx, 1.0, initial_w, 2, 0.1)
print(w, loss)

# expected_loss = 0.972165
# expected_w = np.array([[0.216062], [0.467747]])
