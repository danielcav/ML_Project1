import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from helpers import *


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def reduce_outliers(x):
    """
    Reduce outliers of a vector using the mean and standard deviation method.
    An outlier is defined as following : outlier = ¦x_value¦ > mean + 2 * (standard deviation).
    Then the value is assigned as following : outlier = mean + 2 * (standard deviation).
    """
    mean = np.mean(np.squeeze(x))
    std = np.std(np.squeeze(x))
    threshold = mean + 2 * std
    x_ = (x - mean) / std
    for position, value in enumerate(x_):
        if np.abs(value) > threshold:
            x_[position] = threshold
    return x_

    x_clean = [mean if np.abs(val) > threshold else x for val in x]
    return x_clean


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
    y, tx, ids = load_csv_data(data_path="data/train.csv")
    if standard:
        tx, x_mean, x_std = standardize(tx)
        y, y_mean, y_std = standardize(y)
    return y, tx


def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / y.shape[0]
    return grad


def compute_mse(y, tx, w):
    err = y - tx.dot(w)
    err[err > 1e150] = 1e150
    err[err < -1e150] = -1e150
    return 0.5 * np.linalg.norm(err)**2/y.shape[0]


def sigmoid(t):
    """apply sigmoid function on t."""
    t[t < -700] = -700
    return 1.0 / (1 + np.exp(-t))


def compute_logistic_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.shape[0]
    return grad


def compute_logistic_loss(y, tx, w):
    pred = tx.dot(w)
    pred[pred > 700] = 700  # justifier ca, pourquoi 700 et pas une autre valeur
    loss = np.sum(np.log(1 + np.exp(pred)) - y*pred) / y.shape[0]
    return loss


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
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Parameters
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    n = y.shape[0]
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
        loss = compute_logistic_loss(y, tx, w) + 0.5 * (lambda_ / n) * np.squeeze(w.T.dot(w))
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]
