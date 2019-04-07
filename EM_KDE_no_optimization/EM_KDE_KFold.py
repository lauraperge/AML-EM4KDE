import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn import model_selection

from utils import plot_normal, e_step, m_step

## Load data
data = loadmat('../faithfull/faithful.mat')['X']

data = data[:100]  # taking only a small part for testing

num_data, dim = data.shape

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=False)
CV_Split = CV.split(data)

## Loop until you're happy
epsilon = 1e-4
sigma = np.eye(dim)
log_likelihood = np.asarray([])
i = 0
while True:
    i += 1
    sigmas = []
    # for idx, x_test in enumerate(data): 
    for train_index, test_index in CV_Split:
        # extract training and test set for current CV fold
        x_train = data[train_index, :]
        x_test = data[test_index, :]

        # E step
        responsibility = e_step(x_test, x_train, sigma)
        # print(responsibility.shape)
        
        # M step
        sigmas.append(m_step(x_test, x_train, responsibility, dim))
        
    #sigmas = np.asarray(sigmas).reshape([K, -1, dim, dim])
    sigmas = np.array(sigmas).reshape((K, -1, dim, dim))
    # sum according to num_train, sum according to fold, but note doesnt say to divide by K
    sigma = sigmas.sum(axis=1).sum(axis=0)  # / K

    _log_likelihood = np.zeros(K)
    idx = 0
    print('hey')
    for train_index, test_index in CV_Split:
        # extract training and test set for current CV fold
        x_train = data[train_index, :]
        x_test = data[test_index, :]
        print('ho')
        L = 0
        for train in x_train:
            for test in x_test:
                print('go')
                L += multivariate_normal.pdf(test, mean=train, cov=sigma)
        _log_likelihood[idx] = np.sum(np.log(L))
        idx += 1
    #print(_log_likelihood)
    log_likelihood = np.append(log_likelihood, _log_likelihood.sum())
    
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
