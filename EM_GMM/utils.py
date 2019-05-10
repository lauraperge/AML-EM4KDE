import random

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def is_converged(log_likelihood, epsilon):
    i = len(log_likelihood)
    if i > 1:
        change = log_likelihood[-1] / log_likelihood[-2] - 1.
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))

        if log_likelihood[-1] < log_likelihood[-2]:
            print('Error: Log likelihood decreases.')
            return True

        if abs(change) < epsilon:
            print('Finished.')
            return True
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))
        return False


def remove_random_value(data_array):
    num_data, dim = data_array.shape
    removed_values = []

    def remove_random(item):
        i = round(random.random() * dim - 1)
        removed_values.append(item[i])
        item[i] = None
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


## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
    l, V = np.linalg.eigh(Sigma)
    l[l < 0] = 0
    t = np.linspace(0.0, 2.0*np.pi, 100)
    xy = np.stack((np.cos(t), np.sin(t)))
    Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
    plt.plot(Txy[:, 0], Txy[:, 1])


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

   
