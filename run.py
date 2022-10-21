from implementations import *


# y, tx = load_data(True)
# print(y.max(), np.sort(tx.max(axis=1))[-6:-1])
# initial_w = np.ones(30)
initial_w = np.array([[0.5], [1.0]])
#initial_w = np.array([[0.463156], [0.939874]])
y = np.array([[0.1], [0.3], [0.5]])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
y = (y > 0.2) * 1.0
w, loss = logistic_regression(y, tx, initial_w, 2, 0.1)
print(w, loss)

expected_loss = 1.348358