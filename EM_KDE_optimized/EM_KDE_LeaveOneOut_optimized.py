import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal

from lori.plot import plot_kde
from EM_KDE_optimized.utils import e_step, m_step

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

    R = np.linalg.cholesky(sigma)
    A = data.dot(np.linalg.inv(R).T)
    for idx, x_test in enumerate(data):
        x_train = np.concatenate((data[:idx], data[idx + 1:]), axis=0)

        a_train = np.concatenate((A[:idx], A[idx + 1:]), axis=0)
        a_test = A[idx]

        # E step
        responsibility = e_step(a_test, a_train, R)

        # M step
        sigmas[idx] = m_step(x_test, x_train, responsibility, dim)

    sigma = sigmas.sum(axis=1).mean(axis=0)

    _log_likelihood = np.zeros(num_data)
    pi = 1.0 / (num_data - 1)
    for idx, x_test in enumerate(data):
        x_train = np.concatenate((data[:idx], data[idx + 1:]), axis=0)
        L = 0
        for train in x_train:
            L += pi * multivariate_normal.pdf(x_test, mean=train, cov=sigma)
        _log_likelihood[idx] = np.sum(np.log(L))
    log_likelihood = np.append(log_likelihood, _log_likelihood.sum())

    # sigma = sigmas.sum(axis=1).mean(axis=0)
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

plt.figure(2)
## Plot data
plot_kde(data, sigma, 0.1)
