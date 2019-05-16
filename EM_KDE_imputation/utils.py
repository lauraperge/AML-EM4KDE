import random

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def e_step(a_test, a_train, R):
    num_test, num_train = len(a_test), len(a_train)

    responsibility = np.zeros([num_train, num_test])
    pi = 1.0 / num_train

    for k, train in enumerate(a_train):
        for j, test in enumerate(a_test):
            responsibility[k, j] = pi * custom_normal_pdf(test, mean=train, R=R)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility


def m_step(x_test, x_train, responsibility):
    num_train, dim = x_train.shape
    num_test = len(x_test)

    sigmas = np.zeros([num_train, dim, dim])

    for k, train in enumerate(x_train):
        _sigmas = np.zeros([num_test, dim, dim])
        for n, test in enumerate(x_test):
            delta = (test - train)[np.newaxis]
            _sigmas[n] = (responsibility[k, n] * delta.T).dot(delta)
        sigmas[k] = _sigmas.mean(axis=0)
    return sigmas


def calculate_log_likelihood(x_test, x_train, R):
    num_test, num_train = len(x_test), len(x_train)

    L = np.zeros([num_test, num_train])
    pi = 1.0 / num_train
    for j, test in enumerate(x_test):
        for k, train in enumerate(x_train):
            L[j, k] = pi * custom_normal_pdf(test, mean=train, R=R)
            # L[j, k] = pi * multivariate_normal.pdf(test, mean=train, cov=R)
    return L.sum()


def is_converged(log_likelihood, epsilon):
    i = len(log_likelihood)
    if i > 1:
        change = log_likelihood[-1] / log_likelihood[-2] - 1.
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))

        if log_likelihood[-1] < log_likelihood[-2]:
            print('Error: Log likelihood decreases.')
            return True

        if change < epsilon:
            print('Finished.')
            return True
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))
        return False


## custom mv. norm pdf
def custom_normal_pdf(a, mean, R):
    """Multivariate Normal (Gaussian) probability density function with custom implementation. 
 
    Parameters 
    ---------------------------------- 
        a : array_like 
            Quantiles, with the last axis of `x` denoting the components. (In our implementation a_n for test data) 
         
        mean : array_like 
            Mean of distribution. (In our implementation a_k for train data) 
         
        cov_lower_triangle : array_like
            Lower triangle matrix of covariance of distribution. (In our implementation R) 
     
    Returns 
    --------------------------------- 
         pdf : ndarray or scalar
            Probability density function evaluated at `a` 
 
    """
    if R.ndim < 2:
        dim = 1
    else:
        dim = R.shape[0]

    PI2R = (2 * np.pi) ** (dim / 2) * np.linalg.det(R)

    if a.ndim == 0:
        a = a[np.newaxis]
    elif a.ndim == 1:
        if dim == 1:
            a = a[:, np.newaxis]
        else:
            a = a[np.newaxis, :]
    # pdf = (1 / PI2R) * (np.exp(- 0.5 * (np.sum(np.power(a, 2), 1) - 2 *
    #                                     a.dot(mean.T) + np.sum(np.power(mean, 2)))))

    distance = a - mean
    pdf = (1 / PI2R) * np.exp(- 0.5 * (distance.dot(distance.T)))

    return np.squeeze(pdf)


def remove_random_value(data_array):
    num_data, dim = data_array.shape
    removed_values = []

    def remove_random(item):
        size = round(random.random() * (dim - 2)) + 1
        idx = np.sort(np.unique(np.random.choice(range(dim), size=size, replace=False)))
        removed_dims = []
        for i in idx:
            removed_dims.append(item[i])
            item[i] = None
        removed_values.append(removed_dims)
        return item

    damaged_data = np.array([remove_random(data) for data in data_array])
    removed_values = np.array(removed_values)

    return [damaged_data, removed_values]


def conditional_expectation(test, mean, sigma, existing_dim, missing_dim):
    S22_inv = np.linalg.inv(sigma[np.ix_(existing_dim, existing_dim)])
    S12 = sigma[np.ix_(missing_dim, existing_dim)]

    m1 = mean[missing_dim]
    m2 = mean[existing_dim]

    return m1 + S12.dot(S22_inv.dot(test - m2))


def nadaraya_watson_imputation(damaged_data, train_data, sigma):
    # get indexes of the missing and existing dimensions of the test data
    missing_dim = [idx for idx, value in enumerate(damaged_data) if np.isnan(value)]
    existing_dim = [idx for idx, value in enumerate(damaged_data) if not np.isnan(value)]

    # Remove the placeholder 'nan' values from the damaged data
    damaged_data = damaged_data[np.ix_(existing_dim)]

    # create sigma values for the missing and existing dimensions
    existing_dim_sigma = sigma[np.ix_(existing_dim, existing_dim)]

    train_existing = np.delete(train_data, missing_dim, axis=1)
    train_missing = np.delete(train_data, existing_dim, axis=1)

    # imputed_values = np.sum(train_missing * probabilities[:, np.newaxis], axis=0) / prob_sum

    # create transformed data
    R = np.linalg.cholesky(existing_dim_sigma)
    R_inv_T = np.linalg.inv(R).T
    a_train = train_existing.dot(R_inv_T)
    a_test = damaged_data.dot(R_inv_T)

    responsibility = np.squeeze(e_step(np.array([a_test]), a_train, R))

    imputed_values = np.sum((train_missing * responsibility[:, np.newaxis]), axis=0)

    return imputed_values


def improved_nadaraya_watson_imputation(damaged_data, train_data, sigma):
    # get indexes of the missing and existing dimensions of the test data
    missing_dim = np.array([idx for idx, value in enumerate(damaged_data) if np.isnan(value)])
    existing_dim = np.array([idx for idx, value in enumerate(damaged_data) if not np.isnan(value)])

    # Remove the placeholder 'nan' values from the damaged data
    damaged_data = damaged_data[np.ix_(existing_dim)]

    # create sigma values for the missing and existing dimensions
    existing_dim_sigma = sigma[np.ix_(existing_dim, existing_dim)]

    train_existing = np.delete(train_data, missing_dim, axis=1)

    # create transformed data
    R = np.linalg.cholesky(existing_dim_sigma)
    R_inv_T = np.linalg.inv(R).T
    a_train = train_existing.dot(R_inv_T)
    a_test = damaged_data.dot(R_inv_T)

    responsibility = np.squeeze(e_step(np.array([a_test]), a_train, R))

    cond_exp = np.array(
        [conditional_expectation(damaged_data, mean, sigma, existing_dim, missing_dim) for mean in train_data])

    imputed_values = np.sum((cond_exp * responsibility[:, np.newaxis]), axis=0)

    return imputed_values
