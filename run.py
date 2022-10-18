from implementations import *



y, tx = load_data()
initial_w = np.ones(30)
w, loss = mean_squared_error_sgd(y, tx, initial_w, 100, 0.01)

print(w, loss)
