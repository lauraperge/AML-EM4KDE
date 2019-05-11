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
        idx = np.random.choice(range(dim), size=2, replace=False)
        removed_dims = []
        for i in idx:
            removed_dims.append(item[i])
            item[i] = None
        removed_values.append(removed_dims)
        return item

    damaged_data = np.array([remove_random(data) for data in data_array])
    removed_values = np.array(removed_values)

    return [damaged_data, removed_values]


def remove_dim(sigma, dim):
    reduced_sigma = np.delete(sigma, dim, axis=0)
    reduced_sigma = np.delete(reduced_sigma, dim, axis=1)
    return reduced_sigma


def conditional_expectation(mean, test, sigma, dim):
    # S11 = remove_dim(sigma, dim)
    S22_inv = 1 / sigma[dim][dim]
    S12 = np.delete(sigma[dim], dim, axis=0)[np.newaxis].T

    m1 = mean[dim]
    m2 = np.delete(mean, dim, axis=0)

    return np.squeeze(m1 + S12.dot(S22_inv * (test - m2)))


def nadaraya_watson_imputation(damaged_data, train_data, sigma):
    print(damaged_data)
    missing_dim = [idx for idx, value in enumerate(damaged_data) if np.isnan(value)]
    existing_dim = [idx for idx, value in enumerate(damaged_data) if not np.isnan(value)]

    damaged_data = damaged_data[np.ix_(existing_dim)]

    # remove data of that dimension
    reduced_sigma = sigma[np.ix_(existing_dim, existing_dim)]
    misssing_sigma = sigma[np.ix_(missing_dim, missing_dim)]

    # create transformed data
    R_reduced = np.linalg.cholesky(reduced_sigma)
    R_reduced_inv_T = np.linalg.inv(R_reduced).T

    R_missing_T = np.linalg.cholesky(misssing_sigma).T

    R = np.linalg.cholesky(sigma)
    R_inv_T = np.linalg.inv(R).T
    a = train_data.dot(R_inv_T)
    a_train_set = np.delete(a, missing_dim, axis=1)
    a_train_missing = np.delete(a, existing_dim, axis=1)
    a_test = damaged_data.dot(R_reduced_inv_T)

    probabilities = np.array(
        [np.array(custom_normal_pdf(a_test, mean=a_train, R=R_reduced)) for a_train in a_train_set])

    a_imputed_values = np.sum(a_train_missing * probabilities[:, np.newaxis], axis=0) / np.sum(probabilities)

    imputed_values = a_imputed_values.dot(R_missing_T)

    # !!!! imputed values need to be rebased !!!!

    return imputed_values


if __name__ == '__main__':
    ## Load data
    data = loadmat('../faithfull/faithful.mat')['X']
    data = data  # taking only a small part for testing

    num_data, dim = data.shape
    sigma = np.asarray([[0.28570579, 0.03680529],
                        [0.03680529, 1.0997311]])
    R = np.linalg.cholesky(sigma)

    A = data.dot(np.linalg.inv(R).T)

    print(A.shape, data.shape)

    x_test = data[0]
    x_train = data[1:]

    a_test = A[0]
    a_train = A[1:]

    scipy_norm = []
    custom_norm = []
    for train in a_train:
        custom_norm.append(custom_normal_pdf(a=a_test, mean=train, R=R))

    # for train in x_train:
    #     custom_norm.append(pdf(x=x_test, mean=train, sigma=sigma))

    for train in x_train:
        scipy_norm.append(multivariate_normal.pdf(x=x_test, mean=train, cov=sigma))
