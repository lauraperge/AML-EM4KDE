import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn import model_selection

from EM_KDE_no_optimization.utils import e_step, m_step
from EM_KDE_no_optimization.plot import plot_kde

## Load data
data = loadmat('../faithfull/faithful.mat')['X']

## Real world data (may make sense to crop end, since it's quite big)
# data = np.genfromtxt('../data/winequality-white.csv', delimiter=';')[1:,:]

data = data[:250]  # taking only a small part for testing

num_data, dim = data.shape

# K-fold crossvalidation
K = 250
CV = model_selection.KFold(n_splits=K, shuffle=True)

## Loop until you're happy
epsilon = 1e-3
sigma = np.eye(dim)
log_likelihood = np.asarray([])
i = 0
while True:
    i += 1
    sigmas = []

    for train_index, test_index in CV.split(data):
        # extract training and test set for current CV fold
        x_train = data[train_index, :]
        x_test = data[test_index, :]

        # E step
        responsibility = e_step(x_test, x_train, sigma)

        # M step
        sigmas.append(m_step(x_test, x_train, responsibility, dim))

    sigmas = np.array(sigmas)

    # sum according to num_train, sum according to fold, but note doesnt say to divide by K
    sigma = sigmas.sum(axis=1).mean(axis=0)

    # calculate log likelihood
    _log_likelihood = np.zeros(K)
    idx = 0
    for train_index, test_index in CV.split(data):
        # extract training and test set for current CV fold
        x_train = data[train_index, :]
        x_test = data[test_index, :]

        num_test, num_train = len(x_test), len(x_train)

        L = np.zeros([num_test, num_train])
        pi = 1.0 / num_train
        for j, test in enumerate(x_test):
            for k, train in enumerate(x_train):
                L[j, k] = pi * multivariate_normal.pdf(test, mean=train, cov=sigma)
        _log_likelihood[idx] = L.sum()

        idx += 1
    log_likelihood = np.append(log_likelihood, _log_likelihood.mean())

    if i > 1:
        if log_likelihood[-1] < log_likelihood[-2]:
            print('Error: Log likelihood decreases.')
            break

        change = log_likelihood[-1] / log_likelihood[-2] - 1.
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))
        if change < epsilon:
            break
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))

plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.show()

plot_kde(data, sigma, 0.1)
