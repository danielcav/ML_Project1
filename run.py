from implementations import *


y, tx = load_data(True)
# print(y.max(), np.sort(tx.max(axis=1))[-6:-1])
initial_w = np.ones(30)

loss = compute_logistic_loss(y, tx, initial_w)
print(loss)
