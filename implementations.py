import numpy as np
import matplotlib.pyplot as plt
from helpers import *



# Necessary functions for coding of required methods

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



# Required functions to be implemented


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


# Functions to process data

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



# Extra functions for better results


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

            
def jet_num(y, tx, cat):
    """
    Separate data according to category of value in PRI_jet_num variable (col. 23 in original data set). 
    Three arrays with corresponding classes (either = 0, = 1 or >= 2)
    
    Parameter
    ---------
    tx : ndarray, samples and features
    y : ndarray, actual predictions
    cat : ndarray, categories (jet number) of each sample
    
    Return
    ------
    tx_jet_num : list of three ndarrays, samples and features separated according to category
    y_  : list of three ndarrays, predictions separated according to category
    """
    tx_0 = np.copy(tx)
    tx_1 = np.copy(tx)
    tx_2 = np.copy(tx)
    
    tx_0[cat != 0, :] = np.zeros(tx_0.shape[1])
    tx_1[cat != 1, :] = np.zeros(tx_1.shape[1])
    tx_2[cat < 2, :] = np.zeros(tx_2.shape[1])
    
    y_0 = np.copy(y)
    y_1 = np.copy(y)
    y_2 = np.copy(y)
    
    y_0[cat != 0] = 0
    y_1[cat != 1] = 0
    y_2[cat < 2] = 0
    
    tx_jet_num = [tx_0, tx_1, tx_2]
    y_ = [y_0, y_1, y_2]
    
    return y_, tx_jet_num
            



# Useful methods to run simulation

def train(y, tx, best_lambda, best_degree):
    """
    Train Higgs boson data following multiple steps detailed in code such as data processing, polynomial feature expansion and data splitting based on categories of jet

    Parameters
    ----------
    tx : ndarray, samples and features
    best_lambda : float, regularization parameter
    best_degree : int, degree for feature expansion
    y : ndarray, actual predictions

    Return
    ------
    weights : list of ndarrays, all weights for each category
    losses : list of ndarrays, minimized loss for each category
    """
    #Store column of PRI_jet_num variable
    tx_jet = tx[:,22]
    
    #Data processing
    tx_train, to_replace = data_processing(y, tx)
    
    # Initialization
    features = np.arange(0, tx_train.shape[1], 1)
    
    # Polynomial expansion on training data
    poly_tx_train = polynomial_features(tx_train, features, best_degree)
    
    #Data split according to jet category
    Y_, tx_jet_num = jet_num(y, poly_tx_train, tx_jet)
    
    # To store weights and losses for each sub-sample
    weights = []
    losses = []

    # Weights calculation for each category using ridge regression

    for i in range(len(tx_jet_num)):
        w, loss = ridge_regression(Y_[i], tx_jet_num[i], best_lambda)
        weights.append(w)
        losses.append(loss)
        
    return weights, losses


def test(tx_test, best_lambda, best_degree, weights):
    """
    Test Higgs boson data with calculated weights following multiple steps detailed in code such as data processing, polynomial feature expansion and data splitting based on categories of jet

    Parameters
    ----------
    tx : ndarray, samples and features to be tested
    best_lambda : float, regularization parameter
    best_degree : int, degree for feature expansion
    weights : list of ndarrays, weights for each category

    Return
    ------
    Y_pred : ndarray, all predictions for each category
    """
    #Store column of PRI_jet_num variable
    tx_test_jet = tx_test[:,22]
    
    #Data processing
    X_test, to_replace = data_processing_test(tx_test)
    
    # Initialization
    features = np.arange(0, X_test.shape[1], 1)
    
    # Polynomial expansion on training data
    poly_tx_test = polynomial_features(X_test, features, best_degree)
    
    # Data split according to jet category
    # We set the input y to zeros because we don't know it
    z = np.zeros(X_test.shape[0])
    _, tx_test_jet_num = jet_num(z, poly_tx_test, tx_test_jet)
    
    # List to store the predictions of each category
    pred = []

    #Prediction calculation and storage for each sub-sample

    for i in range(len(tx_test_jet_num)):
        y_pred = np.dot(tx_test_jet_num[i], weights[i])
        pred.append(y_pred)
    
    # Add all predictions to have the final one (before changing to -1 or 1 values)
    Y_pred = np.zeros(tx_test.shape[0])
    for i in range(len(pred)):
        Y_pred += pred[i]
        
    return Y_pred



# Useful method to label predictions with 1 and -1 given a threshold of 0.5

def predict_labels_split(y_pred):
    """Generates class predictions given a y"""
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred