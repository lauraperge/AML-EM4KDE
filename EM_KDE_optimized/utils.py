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

## E-step of the EM algo
def e_step(x_test, x_train, sigma):
    num_test, num_train = len(x_test), len(x_train)
    pi = 1.0 / num_train
    responsibility = np.zeros([num_train])
    for k, train in enumerate(x_train):
        responsibility[k] = pi * \
            multivariate_normal.pdf(x_test, mean=train, cov=sigma)
    responsibility /= np.sum(responsibility, axis=0)

    return responsibility

## M-step of the EM-algo
def m_step(x_test, x_train, responsibility, dim):
    num_test, num_train = len(x_test), len(x_train)

    sigmas = np.zeros([num_train, dim, dim])

    for k, train in enumerate(x_train):
        delta = (x_test - train)[np.newaxis]
        sigmas[k] = (responsibility[k] * delta.T).dot(delta)
    return sigmas

## custom mv. norm pdf
def custom_normal_pdf(a, mean, cov_lower_triangle):
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
    dim = len(mean)
    
    PI2R = np.power(2*np.pi, dim/2) * np.prod(np.diagonal(cov_lower_triangle))

    if len(a)>1:
        pdf = np.exp(-np.log(PI2R) - 0.5 * (np.sum(np.power(a, 2), 1) - 2 *
                                            a.dot(mean.T) + np.sum(np.power(mean, 2))))
    else:
        pdf = np.exp(-np.log(PI2R) - 0.5 * (np.sum(np.power(a, 2)) - 2 *
                                            a.dot(mean.T) + np.sum(np.power(mean, 2))))
    
    return(pdf)


########### try
## Load data
data = loadmat('../faithfull/faithful.mat')['X']
data = data[:20]  # taking only a small part for testing

num_data, dim = data.shape
sigma = np.matrix([[1.2, 2.3], [-0.1, 0.2]])
r = np.linalg.cholesky(sigma)

A = data.dot(np.linalg.inv(r))

print(A.shape, data.shape)

x_test = data[:2]
x_train = data[3:]

a_test = A[:2]
a_train = A[3:]

scipy_norm = []
custom_norm = []
for train in a_train:
    custom_norm.append(custom_normal_pdf(a = a_test, mean = train, cov_lower_triangle = r))

for train in x_train:
    scipy_norm.append(multivariate_normal.pdf(x = x_test, mean = train, cov = sigma))

# print('cust :  {}; scipy:  {}'.format(custom_norm, scipy_norm))

diff = np.array(np.array(custom_norm).reshape((17,2)) - np.array(scipy_norm))
coord = [i for i in range(len(a_train))]

plt.plot(coord, diff[:, 1])
plt.show()
