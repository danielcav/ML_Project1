import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from helpers import *


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def load_data(standard=True):
    # Load data using helpers.py functions and standardize it
    y, tx, ids = load_csv_data(data_path="C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")
    if standard:
        tx, x_mean, x_std = standardize(tx)
        y, y_mean, y_std = standardize(y)
    return y, tx


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def compute_mse(y, tx, w):
    err = y - tx.dot(w)
    return 0.5 * np.mean(err ** 2)


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def compute_logistic_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.shape[0]
    return grad


def compute_logistic_loss(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = (y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))) / y.shape[0]
    return np.squeeze(-loss)


# Implemented functions

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Parameters
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        # update w by gradient descent
        w -= gamma * grad
    return w, compute_mse(y, tx, w)  # last w vector and the corresponding loss


#  Main functions


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Parameters
    """
    w = initial_w
    for n_iters in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1):  # standard mini-batch-size 1
            grad = compute_gradient(batch_y, batch_tx, w)
            # update w by gradient descent
            w -= gamma * grad

    return w, compute_mse(y, tx, w)  # last w vector and the corresponding loss


def least_squares(y, tx):
    """
    Parameters
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)  # use .lstsq or .solve ?
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Parameters
    """
    lambda_prime = 2 * tx.shape[0] * lambda_
    aI = lambda_prime * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Parameters
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma * grad
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Parameters
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    n = y.shape[0]
    for n_iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w) + 0.5 * (lambda_ / n) * np.squeeze(w.T.dot(w))
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, compute_logistic_loss(y, tx, w)
