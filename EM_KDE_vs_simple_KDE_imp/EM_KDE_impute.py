import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection, preprocessing
from sklearn.metrics import mean_squared_error

from utils import remove_random_value, conditional_expectation, e_step, m_step, \
    calculate_log_likelihood, is_converged, nadaraya_watson_imputation
from plot import plot_kde

## Load data
raw_data = preprocessing.normalize(loadmat('../faithfull/wine.mat')['X'])

## Real world data (may make sense to crop end, since it's quite big)
# data = np.genfromtxt('../data/winequality-white.csv', delimiter=';')[1:,:250]

# ## Testing with higher dimension data
# np.random.shuffle(raw_data)
# raw_data = np.concatenate([raw_data, loadmat('../faithfull/faithful.mat')['X']], axis=1)

NUM_TEST = 20

raw_data = raw_data[:1000]  # taking only a small part for testing
data = np.array(raw_data[:-NUM_TEST])
[damaged_data, removed_values] = remove_random_value(raw_data[-NUM_TEST:])
medians = np.median(data, axis=0)  # for baseline

num_data, dim = data.shape

# K-fold cross validation
K = num_data
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

#
# sigma = np.array([[4.28747436e-02, 2.92396851e-01, 2.46394066e-04, 1.05465785e-01],
#                   [2.92396851e-01, 1.44238149e+01, 4.95674770e-02, -1.75754718e+00],
#                   [2.46394066e-04, 4.95674770e-02, 5.51668545e-02, 2.07264980e-01],
#                   [1.05465785e-01, -1.75754718e+00, 2.07264980e-01, 1.57786340e+01]])

# sigma = [[0.0322203, 0.0194771],
#          [0.0194771, 3.8548159]]


# sigma = np.array([[6.48119152e-04, 1.97261524e-05, 3.28995716e-05, 1.75467328e-04,
#                    5.53304683e-06, -3.72432718e-04, -4.21233922e-04, 4.06141951e-05,
#                    1.14670648e-04, 1.89826881e-05, 3.52954365e-04, 2.22300834e-04],
#                   [1.97261524e-05, 8.16157072e-06, -1.54762417e-06, 1.27332809e-05,
#                    5.07504989e-07, -7.58816603e-07, -2.77414572e-05, 3.46213656e-06,
#                    1.20344595e-05, 1.45252595e-06, 3.26173134e-05, 6.50215187e-06],
#                   [3.28995716e-05, -1.54762417e-06, 6.11577717e-06, 1.31000627e-05,
#                    1.91241843e-07, -3.35425225e-05, -1.47965279e-05, 7.34329848e-07,
#                    5.90729211e-07, 1.09060628e-06, 6.54676389e-06, 8.76059867e-06],
#                   [1.75467328e-04, 1.27332809e-05, 1.31000627e-05, 2.44634013e-04,
#                    3.81217214e-06, -1.56352584e-04, -2.47345607e-04, 1.50084065e-05,
#                    4.84157381e-05, 1.03766198e-05, 2.20036688e-04, 7.65131728e-05],
#                   [5.53304683e-06, 5.07504989e-07, 1.91241843e-07, 3.81217214e-06,
#                    2.33206129e-07, -3.28466371e-06, -7.83399152e-06, 7.00015419e-07,
#                    2.33877308e-06, 6.89912373e-07, 7.51433018e-06, 3.87134436e-06],
#                   [-3.72432718e-04, -7.58816603e-07, -3.35425225e-05, -1.56352584e-04,
#                    -3.28466371e-06, 1.91523995e-03, -4.10475438e-04, -7.68195651e-06,
#                    -1.02589085e-05, -4.11535493e-06, -1.15427325e-04, -6.03481937e-05],
#                   [-4.21233922e-04, -2.77414572e-05, -1.47965279e-05, -2.47345607e-04,
#                    -7.83399152e-06, -4.10475438e-04, 8.97538715e-04, -5.38355036e-05,
#                    -1.79734479e-04, -3.28509088e-05, -5.90570324e-04, -3.38682201e-04],
#                   [4.06141951e-05, 3.46213656e-06, 7.34329848e-07, 1.50084065e-05,
#                    7.00015419e-07, -7.68195651e-06, -5.38355036e-05, 5.94088578e-06,
#                    2.01009921e-05, 3.41455695e-06, 5.53657959e-05, 2.72525962e-05],
#                   [1.14670648e-04, 1.20344595e-05, 5.90729211e-07, 4.84157381e-05,
#                    2.33877308e-06, -1.02589085e-05, -1.79734479e-04, 2.01009921e-05,
#                    7.18571328e-05, 1.15095333e-05, 1.92999232e-04, 9.17542248e-05],
#                   [1.89826881e-05, 1.45252595e-06, 1.09060628e-06, 1.03766198e-05,
#                    6.89912373e-07, -4.11535493e-06, -3.28509088e-05, 3.41455695e-06,
#                    1.15095333e-05, 5.93219807e-06, 3.55081410e-05, 2.34873171e-05],
#                   [3.52954365e-04, 3.26173134e-05, 6.54676389e-06, 2.20036688e-04,
#                    7.51433018e-06, -1.15427325e-04, -5.90570324e-04, 5.53657959e-05,
#                    1.92999232e-04, 3.55081410e-05, 6.68918465e-04, 3.09595531e-04],
#                   [2.22300834e-04, 6.50215187e-06, 8.76059867e-06, 7.65131728e-05,
#                    3.87134436e-06, -6.03481937e-05, -3.38682201e-04, 2.72525962e-05,
#                    9.17542248e-05, 2.34873171e-05, 3.09595531e-04, 4.08194717e-04]])

# SIGMA FOR D-KERNEL
# sigma = np.array([[4.66784973e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 7.13859815e-06, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 4.61235791e-06, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.68764666e-04,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    3.33381214e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 8.08115189e-04, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 2.19661809e-04, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.39194558e-06,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    1.88260155e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 7.21498027e-06, 0.00000000e+00, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 2.03868890e-04, 0.00000000e+00],
#                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.04664093e-04]])

imputed_values = []
restored_data = []
median_impute = []
for test_data in damaged_data:
    missing_dim = [idx for idx, value in enumerate(test_data) if np.isnan(value)]

    imputed_value = nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)

    restored_element = np.insert(test_data, missing_dim, imputed_value)
    restored_data.append(restored_element)
    median_impute.append(medians[missing_dim])

    imputed_values.append(imputed_value)

median_impute = np.array(median_impute)
restored_data = np.array(restored_data)
imputed_values = np.array(imputed_values)

divergence = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values) / removed_values]) * 100
divergence_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute) / removed_values]) * 100
mse = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values) ** 2])
mse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute) ** 2])

plt.figure(2)
plt.plot(np.arange(len(divergence)), divergence, '-b', label='Error')
plt.plot(np.arange(len(divergence_median)), divergence_median, '-r', label='Error median')
leg = plt.legend()
plt.xlabel('Index')
plt.ylabel('Imputation error %')
plt.show()

plt.figure(3)
plt.plot(np.arange(len(mse)), mse, '-b', label='MSE')
plt.plot(np.arange(len(mse)), mse_median, '-r', label='MSE median')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Imputation error MSE')
plt.show()

# plot_kde(data, sigma, 0.1)
