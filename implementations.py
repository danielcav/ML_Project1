import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def standardize(x):
    """
    Standardize the original data set using the Z-score normalization. Ignoring NaN values.
    Formula : Z(x) = x - mean / std

    Parameters
    ----------
    x : ndarray, whole data set matrix (Observations x Features)

    Return
    ------
    Returns the standard data set.
    """
    mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0)
    return x / std_x


def min_max_scaling(x):
    """
    Min-Max scaling normalization method. The data is scaled to a fixed [0,1] range.

    Parameters
    ----------
    x : ndarray, whole data set matrix (Observations x Features)

    Return
    ------
    Returns the normalized data set.
    """
    x = (x - np.nanmin(x, axis=0)) / (np.nanmax(x, axis=0) - np.nanmin(x, axis=0))
    return x


def reduce_outliers(x, mean_val=True):
    """
    Reduce outliers of a matrix using the mean and standard deviation method.
    An outlier is defined as following : outlier = |x_value| > mean + 1.5 * (standard deviation).
    If mean_val is true, assigning the mean value to the outliers. If false, assigning the median value.

    Parameters
    ----------
    x : ndarray, whole data set matrix (Observations x Features)
    mean_val : boolean, define which value will be assigned to the outliers

    Return
    ------
    Returns the data set without outliers.
    """
    for col in x.T:
        mean = np.nanmean(col)
        median = np.nanmedian(col)
        std = np.nanstd(x)
        threshold = mean + 1.5 * std
        if mean_val:
            col[np.where(np.abs(col) > threshold)] = mean
        else:
            col[np.where(np.abs(col) > threshold)] = median
    return x


def rank(x):
    """
    Compute the rank of a vector using ordinal ranking method.
    Parameters
    ----------
    x : ndarray, a vector.

    Return
    ------
    Returns the ranks of the values of the vector, starting with 1.
    """
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return ranks+1


def spearman(x, y):
    """
    Spearman correlation coefficient between two features.
    In our case, better than Pearson correlation because we don't know the distribution of the data.
    Spearman correlation is less sensible to outliers.
    Parameters
    ----------
    x : ndarray, feature
    y : ndarray, feature

    Return
    ------
    Returns the rounded Spearman coefficient.
    """
    cov = np.corrcoef([rank(x), rank(y)])
    return np.round(cov[0, 1], 1)


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

            
def jet_num(y, tx, cat):
    """
    Separate data according to category of value in PRI_jet_num variable (col. 22). 
    Three arrays with corresponding classes (either = 0, = 1 or >= 2)
    
    Parameter
    ---------
    tx : ndarray, samples and features
    
    Return
    ------
    tx_jet_num : list of three ndarrays, samples and features separated according to category
    """
    tx_0 = np.asarray(tx[cat == 0, :])
    tx_1 = np.asarray(tx[cat == 1, :])
    tx_2 = np.asarray(tx[cat >= 2, :])
    
    #tx_0 = np.delete(tx_0, 22, axis=1)
    #tx_1 = np.delete(tx_1, 22, axis=1)
    #tx_2 = np.delete(tx_2, 22, axis=1)
    
    y_0 = np.asarray(y[cat == 0])
    y_1 = np.asarray(y[cat == 1])
    y_2 = np.asarray(y[cat >= 2])
    
    tx_jet_num = [tx_0, tx_1, tx_2]
    y_ = [y_0, y_1, y_2]
    
    return y_, tx_jet_num
            

def compute_gradient(y, tx, w):
    """
    Compute the gradient for the gradient descent method.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    w : ndarray, weights

    Return
    ------
    Returns the computed gradient.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / y.shape[0]
    return grad


def compute_mse(y, tx, w):
    """
    Compute the mean squared error cost.
    Fixed limits of values to avoid overflow.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    w : ndarray, weights

    Return
    ------
    Returns the computed loss.
    """
    err = y - tx.dot(w)
    err[err > 1e150] = 1e150
    err[err < -1e150] = -1e150
    return 0.5 * np.linalg.norm(err)**2/y.shape[0]


def sigmoid(t):
    """
    Apply sigmoid function on t.
    """
    t[t < -700] = -700
    return 1.0 / (1 + np.exp(-t))


def compute_logistic_gradient(y, tx, w):
    """
    Compute gradient for the logistic regression method.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    w : ndarray, weights

    Return
    ------
    Returns the computed gradient.
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.shape[0]
    return grad


def compute_logistic_loss(y, tx, w):
    """
    Compute loss for the logistic regression method.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    w : ndarray, weights

    Return
    ------
    Returns the computed loss.
    """
    pred = tx.dot(w)
    pred[pred > 700] = 700
    loss = np.sum(np.log(1 + np.exp(pred)) - y*pred) / y.shape[0]
    return loss


def hessian(y, tx, w):
    """
    Compute the Hessian matrix.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    w : ndarray, weights

    Return
    ------
    Returns the computed hessian matrix.
    """
    pred = sigmoid(tx.dot(w))
    s = pred*(1-pred)
    s_diag = np.diag(s)
    h = (tx.T.dot(s_diag)).dot(tx) / y.shape[0]
    return h


def data_processing(y, tx):
    """
    Data processing applied to all our data before running algorithms.
    Several steps are better described in the code :
    - Feature correlations scores.
    - Outliers and missing values processing.
    - Data normalization.
    - Value assignation to missing values.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features

    Return
    ------
    x_test : ndarray, processed data
    to_remove : indexes of missing features
    """
    # Generating a correlation matrix
    spearman_matrix = np.zeros((tx.shape[1], tx.shape[1]))
    for ind1, x1 in enumerate(tx.T):
        for ind2, y1 in enumerate(tx.T):
            spearman_matrix[ind1, ind2] = spearman(x1, y1)
    matrix = np.tril(spearman_matrix, -1)
    correlated = np.argwhere(matrix == 1)

    # Counting the ratio of missing values per feature
    ratios = []
    for feature in tx.T:
        ratio = np.count_nonzero(feature == -999) / tx.shape[0]
        ratios.append(ratio)

    # Removing correlated features (with correlation score = 1)
    # To choose the feature with the most information we use the ratio of missing
    # values and remove the feature with the biggest ratio (less information, we assume).
    to_remove = set()
    for position in correlated:
        if ratios[position[0]] >= ratios[position[1]]:
            to_remove.add(position[0])
        else:
            to_remove.add(position[1])

    x_clean = np.delete(tx, list(to_remove), axis=1)

    # Process unusable values and outliers
    x_clean[np.where(x_clean == -999)] = np.nan  # Removing unusable values for better data normalization
    x_clean = reduce_outliers(x_clean)  # Assigning mean or median values to outliers

    # Standardize data using either Z-score or Min-Max method
    # Do this before assigning values to nan values -> better standardization
    x2 = min_max_scaling(x_clean)  # Better when we don't know the distribution of the features

    # Assigning values to nans
    # We calculate the median of all values that correspond to one type of y or the other.
    # Then, we replace nans that correspond to one type of y with the median of the same type.
    s = np.argwhere(y == 1)
    b = np.argwhere(y == -1)
    xt = x2.T
    for index, feature in enumerate(xt):
        med1 = np.nanmedian(feature[s])
        med2 = np.nanmedian(feature[b])
        for s_id in s:
            if np.isnan(feature[s_id]):
                feature[s_id] = med1
        feature[np.isnan(feature)] = med2
        xt[index] = feature
    x_test = xt.T
    return x_test, to_remove


def data_processing_test(tx):
    """
    Data processing applied to testing data before running algorithms.
    Several steps are better described in the code :
    - Feature correlations scores.
    - Outliers and missing values processing.
    - Data normalization.
    - Value assignation to missing values.

    Parameters
    ----------
    tx : ndarray, samples and features

    Return
    ------
    x_test : ndarray, processed data
    to_remove : indexes of missing features
    """
    # Generating a correlation matrix
    spearman_matrix = np.zeros((tx.shape[1], tx.shape[1]))
    for ind1, x1 in enumerate(tx.T):
        for ind2, y1 in enumerate(tx.T):
            spearman_matrix[ind1, ind2] = spearman(x1, y1)
    matrix = np.tril(spearman_matrix, -1)
    correlated = np.argwhere(matrix == 1)

    # Counting the ratio of missing values per feature
    ratios = []
    for feature in tx.T:
        ratio = np.count_nonzero(feature == -999) / tx.shape[0]
        ratios.append(ratio)

    # Removing correlated features (with correlation score = 1)
    # To choose the feature with the most information we use the ratio of missing
    # values and remove the feature with the biggest ratio (less information, we assume).
    to_remove = set()
    for position in correlated:
        if ratios[position[0]] >= ratios[position[1]]:
            to_remove.add(position[0])
        else:
            to_remove.add(position[1])

    x_clean = np.delete(tx, list(to_remove), axis=1)

    # Process unusable values and outliers
    x_clean[np.where(x_clean == -999)] = np.nan  # Removing unusable values for better data normalization
    x_clean = reduce_outliers(x_clean)  # Assigning mean or median values to outliers

    # Standardize data using either Z-score or Min-Max method
    # Do this before assigning values to nan values -> better standardization
    x2 = min_max_scaling(x_clean)  # Better when we don't know the distribution of the features

    # Assigning values to nans
    # We calculate the median of all values.
    # Then, we replace nans with the median of the feature.
    xt = x2.T
    for index, feature in enumerate(xt):
        med = np.nanmedian(feature)
        feature[np.isnan(feature)] = med
        xt[index] = feature
    x_test = xt.T
    return x_test, to_remove

def polynomial_features(x, features, degree=2):
    """
    Polynomial expansion without cross-interactions (such as feature1 * feature2).
    Parameters
    ----------
    x : ndarray, features with samples
    features : ndarray, indexes of features to expand
    degree : int, maximum degree of the polynomial expansion

    Return
    ------
    Returns the augmented data.
    """
    degrees = np.arange(2, (degree+1), 1)
    #
    x_aug = np.column_stack((x, np.ones(x.shape[0])))
    for d in degrees:
        x_aug = np.column_stack(([x_aug, x[:, features] ** d]))
    return x_aug
# Implemented functions


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent and mean squared error cost function.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    initial_w : ndarray, weights
    max_iters : int, number of steps to run
    gamma : float, step size

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
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
    Linear regression using stochastic gradient descent and mean squared error cost function.
    Batch size : 1.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    initial_w : ndarray, weights
    max_iters : int, number of steps to run
    gamma : float, step size

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
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
    Least squares regression using normal equations

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    lambda_ : float, regularization parameter

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
    """
    lambda_prime = 2 * tx.shape[0] * lambda_
    aI = lambda_prime * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    initial_w : ndarray, weights
    max_iters : int, number of steps to run
    gamma : float, step size

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
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
    Regularized linear regression using gradient descent.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    lambda_ : float, regularization parameter
    initial_w : ndarray, weights
    max_iters : int, number of steps to run
    gamma : float, step size

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
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

# Additional functions


def newton_method(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using Newton method with Hessian matrix.

    Parameters
    ----------
    y : ndarray, predictions
    tx : ndarray, samples and features
    initial_w : ndarray, weights
    max_iters : int, number of steps to run
    gamma : float, step size

    Return
    ------
    w : ndarray, best weights
    loss : float, minimized loss
    """
    w = initial_w
    losses = []
    threshold = 1e-8
    for n_iter in range(max_iters):
        hessian_matrix = hessian(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma * np.linalg.inv(hessian_matrix) * grad
        loss = compute_logistic_loss(y, tx, w)
        # log info
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
