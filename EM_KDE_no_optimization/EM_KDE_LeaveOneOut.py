import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal

from utils import plot_normal


def e_step(x_test, x_train, sigma):
    num_test, num_train = len(x_test), len(x_train)
    pi = 1.0 / num_train
    responsibility = np.zeros([num_train])
    for k, train in enumerate(x_train):
        responsibility[k] = pi * \
            multivariate_normal.pdf(x_test, mean=train, cov=sigma)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility


def m_step(x_test, x_train, responsibility):
    num_test, num_train = len(x_test), len(x_train)
    _, dim = x_train.shape

    sigmas = np.zeros([num_train, dim, dim])

    for k, train in enumerate(x_train):
        delta = (x_test - train)[np.newaxis]
        sigmas[k] = (responsibility[k] * delta.T).dot(delta)
    return sigmas


## Load data
data = loadmat('../faithfull/faithful.mat')['X']

data = data[:100]  # taking only a small part for testing

num_data, dim = data.shape

## Loop until you're happy
epsilon = 1e-4
sigma = np.eye(dim)
log_likelihood = np.asarray([])
i = 0
while True:
    i += 1
    sigmas = np.zeros([num_data, num_data - 1, dim, dim])
    for idx, x_test in enumerate(data):
        # x_test = np.asarray([x_test])
        x_train = np.concatenate((data[:idx], data[idx + 1:]), axis=0)

        # E step
        responsibility = e_step(x_test, x_train, sigma)
        # M step
        sigmas[idx] = m_step(x_test, x_train, responsibility, dim)

    _log_likelihood = np.zeros(num_data)
    pi = 1.0 / (num_data - 1)
    for idx, x_test in enumerate(data):
        # x_test = np.asarray([x_test])
        x_train = np.concatenate((data[:idx], data[idx + 1:]), axis=0)
        L = 0
        for train in x_train:
            L += pi * multivariate_normal.pdf(x_test, mean=train, cov=sigma)
        _log_likelihood[idx] = np.sum(np.log(L))
    log_likelihood = np.append(log_likelihood, _log_likelihood.sum())

    sigma = sigmas.sum(axis=1).sum(axis=0) / num_data
    if i > 1:
        change = 1. - log_likelihood[-1] / log_likelihood[-2]
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))
        if change < epsilon:
            break
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))

plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')

## Plot data
plt.figure(2)
if dim == 2:
    plt.plot(data[:, 0], data[:, 1], '.')
if dim == 3:
    plt.plot3(data[:, 0], data[:, 1], data[:, 2], '.')

for _data in data:
    plot_normal(_data, sigma)

plt.show()
