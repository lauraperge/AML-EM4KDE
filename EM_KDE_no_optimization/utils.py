import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def e_step(x_test, x_train, sigma):
    num_test, num_train = len(x_test), len(x_train)

    responsibility = np.zeros([num_train, num_test])
    for k, train in enumerate(x_train):
        responsibility[k] = multivariate_normal.pdf(x_test, mean=train, cov=sigma)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility


def m_step(x_test, x_train, responsibility, dim):
    num_test, num_train = len(x_test), len(x_train)

    sigmas = np.zeros([num_train, dim, dim])

    for k, train in enumerate(x_train):
        _sigmas = np.zeros([num_test, dim, dim])
        for n, test in enumerate(x_test):
            delta = (test - train)[np.newaxis]
            _sigmas[n] = (responsibility[k, n] * delta.T).dot(delta)
        sigmas[k] = _sigmas.mean(axis=0)
    return sigmas
