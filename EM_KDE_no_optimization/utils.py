from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

## Helper function for plotting a 2D Gaussian


def plot_normal(mu, Sigma):
    l, V = np.linalg.eigh(Sigma)
    l[l < 0] = 0
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    xy = np.stack((np.cos(t), np.sin(t)))
    Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
    plt.plot(Txy[:, 0], Txy[:, 1])


def e_step(x_test, x_train, sigma):
    num_test, num_train = len(x_test), len(x_train)
    
    responsibility = np.zeros((num_test, num_train))
    for k, train in enumerate(x_train):
        responsibility[:, k] = multivariate_normal.pdf(x_test, mean=train, cov=sigma)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility


def m_step(x_test, x_train, responsibility, dim):
    num_test, num_train = len(x_test), len(x_train)

    sigmas = np.zeros([num_train, dim, dim])
    sigmas_test = 0
    for k, train in enumerate(x_train):
        for n, test in enumerate(x_test):
            delta = (test - train)[np.newaxis]
            sigmas_test += (responsibility[n, k] * delta.T).dot(delta)
        sigmas[k] = sigmas_test/num_test
    return sigmas
