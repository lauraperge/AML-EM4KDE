import random
import numpy as np
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
            # Use these two lines to reproduce F-kernel
            # delta = (test - train)[np.newaxis]
            # _sigmas[n] = (responsibility[k, n] * delta.T).dot(delta)

            # Use this line to reproduce D-Kernel
            _sigmas[n] = np.eye(dim) * (responsibility[k, n] * (test - train) ** 2)

            # Use this line to reproduce S-Kernel
            # _sigmas[n] = np.eye(dim) * responsibility[k, n] * np.linalg.norm(test-train)

        sigmas[k] = _sigmas.mean(axis=0)

    return sigmas


def calculate_log_likelihood(x_test, x_train, R):
    num_test, num_train = len(x_test), len(x_train)

    L = np.zeros([num_test, num_train])
    pi = 1.0 / num_train
    for j, test in enumerate(x_test):
        for k, train in enumerate(x_train):
            L[j, k] = pi * custom_normal_pdf(test, mean=train, R=R)
    return L.sum()


def is_converged(log_likelihood, epsilon):
    i = len(log_likelihood)
    if i > 1:
        change = log_likelihood[-1] / log_likelihood[-2] - 1.
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))

        # if the Sigma matrix is diagonalized the loglikelihood has a tendency to overshoot
        if i > 2:
            old_change = log_likelihood[-2] / log_likelihood[-3] - 1.
            new_change = log_likelihood[-1] / log_likelihood[-2] - 1.
            if new_change > old_change:
                print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))
                print('Finished.')
                return True

        if log_likelihood[-1] < log_likelihood[-2]:
            print('Error: Log likelihood decreases.')
            return True

        if change < epsilon:
            print('Finished.')
            return True
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))
        return False


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

    distance = a - mean
    pdf = (1 / PI2R) * np.exp(- 0.5 * (distance.dot(distance.T)))

    return np.squeeze(pdf)


def remove_random_value(data_array):
    num_data, dim = data_array.shape
    removed_values = []

    def remove_random(item):
        size = int(round(random.random() * (dim - 2)) + 1)
        # size = 4
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


def remove_dim(sigma, dim):
    reduced_sigma = np.delete(sigma, dim, axis=0)
    reduced_sigma = np.delete(reduced_sigma, dim, axis=1)
    return reduced_sigma


def conditional_expectation(mean, test, sigma, dim):
    S22_inv = 1 / sigma[dim][dim]
    S12 = np.delete(sigma[dim], dim, axis=0)[np.newaxis].T

    m1 = mean[dim]
    m2 = np.delete(mean, dim, axis=0)

    return np.squeeze(m1 + S12.dot(S22_inv * (test - m2)))


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

    probabilities = np.array(
        [np.array(multivariate_normal.pdf(x=damaged_data, mean=train, cov=existing_dim_sigma)) for train in
         train_existing])
    prob_sum = probabilities.sum() if probabilities.sum() != 0 else 1
    imputed_values = np.sum(train_missing * probabilities[:, np.newaxis], axis=0) / prob_sum

    return imputed_values


if __name__ == '__main__':
    a = np.array([5, 3])
    b = np.array([8, 7])
    #
    # print(np.linalg.norm(a-b, ord=2, keepdims=True))
    print(np.eye(2) * (a - b) ** 2)

    delta = (a - b)[np.newaxis]
    print((delta.T).dot(delta))

    # sigma = [[3, 1, -1],
    #          [1, 3, -1],
    #          [-1, -1, 5]]
    #
    # scalarize(sigma)
