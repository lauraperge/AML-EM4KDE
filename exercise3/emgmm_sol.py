import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
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

    sigmas = np.zeros([num_train, dim, dim])

    for k, train in enumerate(x_train):
        delta = (x_test - train)[np.newaxis]
        sigmas[k] = (responsibility[k] * delta.T).dot(delta)
    return sigmas


## Load data
data = loadmat('../faithfull/faithful.mat')['X']

data = data[:50]  # taking only a small part for testing

num_data, dim = data.shape

## Loop until you're happy
epsilon = 0.01  # XXX: you should find a better convergence check than a max iteration counter
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
        sigmas[idx] = m_step(x_test, x_train, responsibility)

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
    print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))

    if i > 50:  # and log_likelihood[-1] / log_likelihood[-2] > epsilon:
        break

## Plot log-likelihood -- did we converge?
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
