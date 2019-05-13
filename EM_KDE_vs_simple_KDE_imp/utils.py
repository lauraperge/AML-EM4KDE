import random

from scipy.linalg import eig
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
            # Use these two lines to reproduce F-kernel, D-kernel_v1 and S-Kernel_v1
            # delta = (test - train)[np.newaxis]
            # _sigmas[n] = (responsibility[k, n] * delta.T).dot(delta)

            # Use this line to reproduce D-Kernel_v2
            _sigmas[n] = np.eye(dim) * (responsibility[k, n] * (test-train)**2)

            # Use this line to reproduce S-Kernel_v2
            # _sigmas[n] = np.eye(dim) * responsibility[k, n] * np.linalg.norm(test-train)

        # Use this line to reproduce F-kernel, D-kernel_v2 and S-Kernel_v2
        sigmas[k] = _sigmas.mean(axis=0)

        # Use this line to reproduce D-Kernel_v1
        # sigmas[k] = diagonalize(_sigmas.mean(axis=0))

        # Use this line to reproduce S-Kernel_v1
        # sigmas[k] = scalarize(_sigmas.mean(axis=0))
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


# def remove_random_value(data_array):
#     num_data, dim = data_array.shape
#     removed_values = []
#
#     def remove_random(item):
#         i = int(round(random.random() * dim - 1))
#         removed_values.append(item[i])
#         item[i] = None
#         return item
#
#     damaged_data = np.array([remove_random(data) for data in data_array])
#     removed_values = np.array(removed_values)
#
#     return [damaged_data, removed_values]

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

# def remove_random_values(data_array):
#     num_data, dim = data_array.shape
#     removed_values = []
#
#     def remove_random(item):
#         # sample from the range(dim)=[0,1,2, ...dim] list random element(s) (1 or 2 or ...dim-2 number of elements)
#         dims_to_remove = random.sample(range(dim), random.randint(1, dim-2))
#         dims_to_remove.sort()
#         for dim_to_remove in dims_to_remove:
#             removed_values.append(item[dim_to_remove])
#             item[dim_to_remove] = None
#         return item
# 
#     damaged_data = np.array([remove_random(data) for data in data_array])
#     removed_values = np.array(removed_values)
#
#     return [damaged_data, removed_values]


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
    # get indexes of the missing and existing dimensions of the test data
    missing_dim = [idx for idx, value in enumerate(damaged_data) if np.isnan(value)]
    existing_dim = [idx for idx, value in enumerate(damaged_data) if not np.isnan(value)]

    # Remove the placeholder 'nan' values from the damaged data
    damaged_data = damaged_data[np.ix_(existing_dim)]

    # create sigma values for the missing and existing dimensions
    existing_dim_sigma = sigma[np.ix_(existing_dim, existing_dim)]
    missing_dim_sigma = sigma[np.ix_(missing_dim, missing_dim)]

    train_existing = np.delete(train_data, missing_dim, axis=1)
    train_missing = np.delete(train_data, existing_dim, axis=1)

    # # create transformed data
    # R = np.linalg.cholesky(sigma)
    # R_reduced = np.linalg.cholesky(existing_dim_sigma)
    # R_missing = np.linalg.cholesky(missing_dim_sigma)
    #
    # a = train_data.dot(np.linalg.inv(R).T)
    # a_train_existing = np.delete(a, missing_dim, axis=1)
    # a_train_missing = np.delete(a, existing_dim, axis=1)
    # a_damaged = damaged_data.dot(np.linalg.inv(R_reduced).T)
    #
    # probabilities = np.array(
    #     [np.array(custom_normal_pdf(a_damaged, mean=a_train, R=R_reduced)) for a_train in a_train_existing])
    #
    # prob_sum = probabilities.sum() or 1  # to avoid dividing by zero
    #
    # a_imputed_values = np.sum(a_train_missing * probabilities[:, np.newaxis], axis=0) / prob_sum
    #
    # imputed_values = a_imputed_values.dot(R_missing.T)

    probabilities = np.array(
        [np.array(multivariate_normal.pdf(x=damaged_data, mean=train, cov=existing_dim_sigma)) for train in
         train_existing])
    prob_sum = probabilities.sum()
    imputed_values = np.sum(train_missing * probabilities[:, np.newaxis], axis=0) / prob_sum

    return imputed_values

##############################################################################
# Diagonalize the given covariance matrix, returning the
# diagonal matrix W, and the unitary matrix V such that
#  V * U * V^{-1} = W

def diagonalize(cov_matrix):
    """Diagonalize a given covariance matrix.

        Parameters
        ----------------------------------
            cov_matrix : array_like
                A covariance matrix where each element contains a non-zero value

        Returns
        ---------------------------------
             matrix_W : array_like
                Reduced covariance matrix where non-zero elements are located along the main axis
                sigma = [[lambda1, 0, 0],[0, lambda2, 0],[0, 0, lambda3]]

        """
    (eig_vals, eig_vecs) = eig(cov_matrix)

    # # Create the diagonalization matrix V
    # matrix_V = np.array(eig_vecs)
    # # Multiply V^{-1} * U * V to diagonalize
    # matrix_W = np.dot(np.linalg.inv(matrix_V),np.dot(cov_matrix, matrix_V))

    # Construct the diagonalized matrix that we want
    matrix_diag = np.array(np.eye(len(cov_matrix)))
    for i in range(len(eig_vals)):
        matrix_diag[(i, i)] = eig_vals[i].real

    # print(matrix_W, matrix_diag)

    return matrix_diag


def scalarize(cov_matrix):
    """Acalarize a given covariance matrix.

        Parameters
        ----------------------------------
            cov_matrix : array_like
                A covariance matrix where each element contains a non-zero value

        Returns
        ---------------------------------
             matrix_W : array_like
                Reduced covariance matrix where non-zero elements are the same and they are located along the main axis
                sigma = [[lambda, 0, 0],[0, lambda, 0],[0, 0, lambda]]


        """
    (eig_vals, eig_vecs) = eig(cov_matrix)

    # Construct the scalarized matrix that we want
    matrix_diag = np.array(np.eye(len(cov_matrix)))
    for i in range(len(eig_vals)):
        matrix_diag[(i, i)] = eig_vals[0].real

    return matrix_diag


if __name__ == '__main__':
    a = np.array([5, 3])
    b = np.array([8, 7])
    #
    # print(np.linalg.norm(a-b, ord=2, keepdims=True))
    print(np.eye(2)*(a-b)**2)

    delta = (a - b)[np.newaxis]
    print((delta.T).dot(delta))

    # sigma = [[3, 1, -1],
    #          [1, 3, -1],
    #          [-1, -1, 5]]
    #
    # scalarize(sigma)
