from implementations import *


y, tx = load_data(False)
initial_w = np.ones(30)
w, loss = logistic_regression(y, tx, initial_w, 100, 0.1)

print(w, loss)
