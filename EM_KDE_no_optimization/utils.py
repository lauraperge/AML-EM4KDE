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
    pi = 1.0 / num_train
    responsibility = np.zeros([num_train])
    for k, train in enumerate(x_train):
        responsibility[k] = pi * multivariate_normal.pdf(x_test, mean=train, cov=sigma)
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
