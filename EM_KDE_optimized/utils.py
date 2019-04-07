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


def e_step(a_test, a_train, R):
    num_test, num_train = len(a_test), len(a_train)

    responsibility = np.zeros([num_train, num_test])
    pi = 1.0 / num_train

    for k, train in enumerate(a_train):
        for j, test in enumerate(a_test):
            responsibility[k, j] = pi * custom_normal_pdf(test, mean=train, R=R)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility


# def e_step(x_test, x_train, sigma):
#     num_test, num_train = len(x_test), len(x_train)
#
#     responsibility = np.zeros([num_train, num_test])
#     for k, train in enumerate(x_train):
#         responsibility[k] = multivariate_normal.pdf(x_test, mean=train, cov=sigma)
#     responsibility /= np.sum(responsibility, axis=0)
#
#     return responsibility


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


def calculate_log_likelihood(x_test, x_train, R):
    num_test, num_train = len(x_test), len(x_train)

    L = np.zeros([num_test, num_train])
    pi = 1.0 / num_train
    for j, test in enumerate(x_test):
        for k, train in enumerate(x_train):
            L[j, k] = pi * custom_normal_pdf(test, mean=train, R=R)
            # L[j, k] = pi * multivariate_normal.pdf(test, mean=train, cov=R)
    return L.sum()


def is_converged(log_likelihood, epsilon):
    i = len(log_likelihood)
    if i > 1:
        change = log_likelihood[-1] / log_likelihood[-2] - 1.
        print('Run {}, log likelihood: {}, change: {}'.format(i, log_likelihood[-1], change))

        if log_likelihood[-1] < log_likelihood[-2]:
            print('Error: Log likelihood decreases.')
            return True

        if change < epsilon:
            print('Finished.')
            return True
    else:
        print('Run {}, log likelihood: {}'.format(i, log_likelihood[-1]))
        return False


def pdf(x, mean, sigma):
    R = np.linalg.cholesky(sigma)
    R_inv = np.linalg.inv(R)

    if sigma.ndim < 2:
        dim = 1
    else:
        dim = sigma.shape[0]

    x = x[np.newaxis].T
    mean = mean[np.newaxis].T

    # print(xT.dot(R_inv.T))
    # print(meanT.dot(R_inv.T))

    PI2R = ((2 * np.pi) ** dim * np.linalg.det(sigma)) ** 0.5

    q_1 = (x.T - mean.T).dot(R_inv.T)
    q_2 = R_inv.dot(x - mean)

    pdf = (1 / PI2R) * np.exp(- 0.5 * (x - mean).T.dot(np.linalg.inv(sigma)).dot(x - mean))
    # pdf = (1 / PI2R) * np.exp(- 0.5 * q_1.dot(q_1.T))

    return np.squeeze(pdf)


## custom mv. norm pdf
def custom_normal_pdf(a, mean, R):
    """Multivariate Normal (Gaussian) probability density function with custom implementation. 
 
    Parameters 
    ---------------------------------- 
        a : array_like 
            Quantiles, with the last axis of `x` denoting the components. (In our implementation a_n for test data) 
         
        mean : array_like 
            Mean of distribution. (In our implementation a_k for train data) 
         
        cov_lower_triangle : array_like
            Lower triangle matrix of covariance of distribution. (In our implementation R) 
     
    Returns 
    --------------------------------- 
         pdf : ndarray or scalar
            Probability density function evaluated at `a` 
 
    """
    if R.ndim < 2:
        dim = 1
    else:
        dim = R.shape[0]

    PI2R = (2 * np.pi) ** (dim / 2) * np.linalg.det(R)
    # PI2R = np.linalg.det(2 * np.pi * R.T.dot(R)) ** 0.5
    if a.ndim == 0:
        a = a[np.newaxis]
    elif a.ndim == 1:
        if dim == 1:
            a = a[:, np.newaxis]
        else:
            a = a[np.newaxis, :]
    # pdf = (1 / PI2R) * (np.exp(- 0.5 * (np.sum(np.power(a, 2), 1) - 2 *
    #                                     a.dot(mean.T) + np.sum(np.power(mean, 2)))))

    distance = a - mean
    pdf = (1 / PI2R) * np.exp(- 0.5 * (
        distance.dot(distance.T)))

    return np.squeeze(pdf)


if __name__ == '__main__':
    ########### try
    ## Load data
    data = loadmat('../faithfull/faithful.mat')['X']
    data = data  # taking only a small part for testing

    num_data, dim = data.shape
    sigma = np.asarray([[0.28570579, 0.03680529],
                        [0.03680529, 1.0997311]])
    R = np.linalg.cholesky(sigma)

    A = data.dot(np.linalg.inv(R).T)

    print(A.shape, data.shape)

    x_test = data[0]
    x_train = data[1:]

    a_test = A[0]
    a_train = A[1:]

    scipy_norm = []
    custom_norm = []
    for train in a_train:
        custom_norm.append(custom_normal_pdf(a=a_test, mean=train, R=R))

    # for train in x_train:
    #     custom_norm.append(pdf(x=x_test, mean=train, sigma=sigma))

    for train in x_train:
        scipy_norm.append(multivariate_normal.pdf(x=x_test, mean=train, cov=sigma))

    print(custom_norm)
    print(scipy_norm)

    diff = np.power(np.subtract(np.array(custom_norm), np.array(scipy_norm)), 2)

    plt.plot(diff)
    plt.show()
