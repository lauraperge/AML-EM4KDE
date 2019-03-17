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


## Load data
# data = loadmat('../lori/faithful.mat')['X']
data = loadmat('clusterdata2d.mat')['data']

data = data[:200]  # taking only a small part for testing

num_data, dim = data.shape

## Loop until you're happy
max_iter = 30  # XXX: you should find a better convergence check than a max iteration counter
sigmas = np.asarray([np.eye(dim) for _ in range(num_data)])

for idx, test in enumerate(data):
    if idx == 0:
        train_array = data[1:]
    if idx == num_data:
        train_array = data[:-1]
    if 0 < idx < num_data:
        train_array = np.concatenate((data[:idx], data[idx + 1:]), axis=0)  # removing the current element

    num_train = train_array.shape[0]

    _sigmas = np.asarray([np.eye(dim) for _ in range(num_train)])
    pi = (1.0 / num_train) * np.ones(num_train)  # same pi for each distribution
    responsibility = np.zeros(num_train)
    log_likelihood = np.zeros(max_iter)

    for iteration in range(max_iter):
        ## Compute responsibilities
        for k, train in enumerate(train_array):
            responsibility[k] = pi[k] * multivariate_normal.pdf(test, mean=train, cov=_sigmas.sum(axis=0))
        responsibility /= np.sum(responsibility, axis=0)

        ## Update parameters
        for k, train in enumerate(train_array):
            responsibility_k = responsibility[k]  # num_train
            Nk = np.sum(responsibility_k)  # scalar
            # mu[k] = responsibility_k.dot(test) / Nk  # dim
            delta = (test - train)[np.newaxis]  # NxD
            _sigmas[k] = (responsibility_k * delta.T).dot(delta)  # / Nk  # dim x dim
            # pi[k] = Nk / num_train

        ## Compute log-likelihood of data
        # log_likelihood[iteration] = -1 # XXX: DO THIS CORRECTLY
        L = 0
        # print(np.linalg.det(_sigmas.sum(axis=0)))
        for k, train in enumerate(train_array):
            L += pi[k] * multivariate_normal.pdf(test, mean=train, cov=_sigmas.sum(axis=0))
        log_likelihood[iteration] = np.sum(np.log(L))
        print(log_likelihood[iteration])
    sigmas[idx] = _sigmas.sum(axis=0)
    print('Run {}/{}'.format(idx + 1, num_data))
    print(sigmas[idx])

sigma = sigmas.mean(axis=0)

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
    print(_data)
    print(sigma)
    plot_normal(_data, sigma)

plt.show()
