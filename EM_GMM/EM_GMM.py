import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal

from utils import remove_random_value, remove_dim, conditional_expectation, is_converged, plot_normal

## Load data
data = loadmat('../faithfull/faithful.mat')['X']

## Real world data (may make sense to crop end, since it's quite big)
# data = np.genfromtxt('../data/winequality-white.csv', delimiter=';')[1:,:250]

## Testing with higher dimension data
# np.random.shuffle(data)
# data = np.concatenate([data, loadmat('../faithfull/faithful.mat')['X']], axis=1)

raw_data = data  # taking only a small part for testing
data = np.array(raw_data[:-10])
[damaged_data, removed_values] = remove_random_value(raw_data[-10:])
medians = np.median(data, axis=0)  # for baseline

num_data, dim = data.shape

## Initialize parameters
K = 3  # try with different parameters
mu = []
Sigma = []
pi_k = np.ones(K)/K

for _ in range(K):
  # Let mu_k be a random data point:
  mu.append(data[np.random.choice(num_data)])
  # Let Sigma_k be the identity matrix:
  Sigma.append(np.eye(dim))

## Loop until convergence
epsilon = 1e-7
log_likelihood = np.asarray([])
respons = np.zeros((K, num_data))  # KxN

while True:

    # E-step
    for k in range(K):
        respons[k] = pi_k[k] * \
            multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
    respons /= np.sum(respons, axis=0)

    # M-step
    for k in range(K):
        respons_k = respons[k]  # N
        Nk = np.sum(respons_k)  # scalar
        mu[k] = respons_k.dot(data) / Nk  # D
        delta = data - mu[k]  # NxD
        Sigma[k] = (respons_k * delta.T).dot(delta) / Nk  # DxD
        pi_k[k] = Nk/num_data

    ## Compute log-likelihood of data
    L = 0
    for k in range(K):
        L += pi_k[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])

    log_likelihood = np.append(log_likelihood, np.sum(np.log(L)))

    if is_converged(log_likelihood, epsilon):
        break

## Plot log-likelihood -- did we converge?
plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.show()

## Plot data (for 2D/3D data)
plt.figure(2)
if dim == 2:
    plt.plot(data[:, 0], data[:, 1], '.')
if dim == 3:
    plt.plot3(data[:, 0], data[:, 1], data[:, 2], '.')

for k in range(K):
    plot_normal(mu[k], Sigma[k])
plt.show()
