import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection

from EM_KDE_optimized.utils import e_step, m_step, calculate_log_likelihood, is_converged
from EM_KDE_optimized.plot import plot_kde

## Load data
data = loadmat('../faithfull/faithful.mat')['X']

## Testing with higher dimension data
# np.random.shuffle(data)
# data = np.concatenate([data, loadmat('../faithfull/faithful.mat')['X']], axis=1)

data = data[:250]  # taking only a small part for testing

num_data, dim = data.shape

# K-fold crossvalidation
K = 50
CV = model_selection.KFold(n_splits=K, shuffle=False)

## Loop until you're happy
epsilon = 1e-3
sigma = np.eye(dim)
log_likelihood = np.asarray([])
i = 0
while True:
    i += 1
    sigmas = []

    R = np.linalg.cholesky(sigma)
    A = data.dot(np.linalg.inv(R).T)

    for train_index, test_index in CV.split(A):
        # extract training and test set for current CV fold
        a_test = A[test_index, :]
        a_train = A[train_index, :]

        x_test = data[test_index, :]
        x_train = data[train_index, :]

        # E step
        responsibility = e_step(a_test, a_train, R)

        # M step
        sigmas.append(m_step(x_test, x_train, responsibility))

    sigma = np.array(sigmas).sum(axis=1).mean(axis=0)

    print(np.linalg.eigvalsh(sigma))

    R = np.linalg.cholesky(sigma)
    A = data.dot(np.linalg.inv(R).T)

    _log_likelihood = []
    for train_index, test_index in CV.split(A):
        # extract training and test set for current CV fold
        x_train = A[train_index, :]
        x_test = A[test_index, :]

        _log_likelihood.append(calculate_log_likelihood(x_test, x_train, R))

    log_likelihood = np.append(log_likelihood, np.asarray(_log_likelihood).mean())

    if is_converged(log_likelihood, epsilon):
        break

plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.show()

plot_kde(data, sigma, 0.1)
