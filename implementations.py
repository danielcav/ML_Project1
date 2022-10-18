import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from helpers import *


def load_data():
    # Load data using helpers.py functions and standardize it
    y_, tx, ids = load_csv_data(data_path="C:/Users/Daniel/OneDrive/Bureau/EPFL/Master/ML/train.csv")
    x, x_mean, x_stx = standardize(tx)
    y, y_mean, y_stx = standardize(y_)
    return y, x


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # linear regression
        err = y - tx.dot(w)
        # compute gradient, loss
        grad = -tx.T.dot(err)/len(err)
        loss = 1 / 2 * np.mean(err ** 2)
        # update w by gradient descent
        w = w - gamma * grad

    return w, loss  # last w vector and the corresponding loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for n_iters in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1):  # standard mini-batch-size 1
            # linear regression
            err = batch_y - batch_tx.dot(w)
            # compute gradient, loss
            grad = -batch_tx.T.dot(err) / len(err)
            loss = 1 / 2 * np.mean(err ** 2)
            # update w by gradient descent
            w = w - gamma * grad

    return w, loss


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    # MSE loss function
    loss = 1 / 2 * np.mean(err ** 2)
    return w, loss


def ridge_regression(y, tx, lambda_):
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    # MSE loss function
    loss = 1 / 2 * np.mean(err ** 2)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        pred = 1.0 / (1 + np.exp(-tx.dot(w)))
        # get loss and update w.
        loss = np.squeeze(-(y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))))
        grad = tx.T.dot(pred - y)
        w -= gamma * grad
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError
