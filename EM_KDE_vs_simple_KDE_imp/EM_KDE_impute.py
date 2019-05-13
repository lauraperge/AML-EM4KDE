import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection, preprocessing
from utils import e_step, m_step, calculate_log_likelihood, is_converged
from plot import plot_kde

@profile
def EM_KDE():
    ## Load data
    raw_data = preprocessing.scale(loadmat('../faithfull/wine.mat')['X'])

    # Taking only small part due to memory limitations
    # Also remove the first 100 what we damage later on purpose
    NUM_TEST = 100
    raw_data = raw_data[:1000]
    data = np.array(raw_data[:-NUM_TEST])
    num_data, dim = data.shape

    # K-fold cross validation
    K = num_data
    CV = model_selection.KFold(n_splits=K, shuffle=False)

    ## Loop until you're happy
    epsilon = 2
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

    print(sigma)

    plt.figure(1)
    plt.plot(log_likelihood)
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.show()


if __name__ == '__main__':
    EM_KDE()