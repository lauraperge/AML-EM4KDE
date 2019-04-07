import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal

from lori.plot import plot_kde
from EM_KDE_optimized.utils import custom_normal_pdf, e_step, m_step

## Load data
data = loadmat('../faithfull/faithful.mat')['X']

data = data[:100]  # taking only a small part for testing

num_data, dim = data.shape

## Loop until you're happy
epsilon = 1e-3
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



    # sigma = sigmas.sum(axis=1).mean(axis=0)
    if i > 1:
        change = (log_likelihood[-1] / log_likelihood[-2]) - 1.
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
