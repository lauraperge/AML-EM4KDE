import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection, preprocessing

from EM_KDE_imputation.utils import remove_random_value, conditional_expectation, e_step, m_step, \
    calculate_log_likelihood, is_converged, nadaraya_watson_imputation, improved_nadaraya_watson_imputation
from EM_KDE_imputation.plot import plot_kde

## Load data
raw_data = loadmat('../faithfull/wine.mat')['X']

NUM_TEST = 100

raw_data = preprocessing.scale(raw_data[:(1000 + NUM_TEST)])  # taking only a small part for testing
data = np.array(raw_data[:-NUM_TEST])
[damaged_data, removed_values] = remove_random_value(raw_data[-NUM_TEST:])
medians = np.median(data, axis=0)  # for baseline

num_data, dim = data.shape

# K-fold cross validation
# K = num_data
# CV = model_selection.KFold(n_splits=K, shuffle=False)
#
# ## Loop until you're happy
# epsilon = 1e-3
# sigma = np.eye(dim)
# log_likelihood = np.asarray([])
# i = 0
# while True:
#     i += 1
#     sigmas = []
#
#     R = np.linalg.cholesky(sigma)
#     A = data.dot(np.linalg.inv(R).T)
#
#     for train_index, test_index in CV.split(A):
#         # extract training and test set for current CV fold
#         a_test = A[test_index, :]
#         a_train = A[train_index, :]
#
#         x_test = data[test_index, :]
#         x_train = data[train_index, :]
#
#         # E step
#         responsibility = e_step(a_test, a_train, R)
#
#         # M step
#         sigmas.append(m_step(x_test, x_train, responsibility))
#
#     sigma = np.array(sigmas).sum(axis=1).mean(axis=0)
#
#     R = np.linalg.cholesky(sigma)
#     A = data.dot(np.linalg.inv(R).T)
#
#     _log_likelihood = []
#     for train_index, test_index in CV.split(A):
#         # extract training and test set for current CV fold
#         x_train = A[train_index, :]
#         x_test = A[test_index, :]
#
#         _log_likelihood.append(calculate_log_likelihood(x_test, x_train, R))
#
#     log_likelihood = np.append(log_likelihood, np.asarray(_log_likelihood).mean())
#
#     if is_converged(log_likelihood, epsilon):
#         break
#
# print(sigma)
#
# plt.figure(1)
# plt.plot(log_likelihood)
# plt.xlabel('Iterations')
# plt.ylabel('Log-likelihood')
# plt.show()

# sigma = np.array([[4.28747436e-02, 2.92396851e-01, 2.46394066e-04, 1.05465785e-01],
#                   [2.92396851e-01, 1.44238149e+01, 4.95674770e-02, -1.75754718e+00],
#                   [2.46394066e-04, 4.95674770e-02, 5.51668545e-02, 2.07264980e-01],
#                   [1.05465785e-01, -1.75754718e+00, 2.07264980e-01, 1.57786340e+01]])
#
# sigma = [[0.0322203, 0.0194771],
#          [0.0194771, 3.8548159]]
#
#
sigma = np.array([[1.81394325e-01, -5.80157882e-04, 1.30766038e-01, -1.64303794e-02,
                   -6.26605536e-03, -2.60816538e-02, -2.65245856e-02, 8.37288402e-06,
                   -1.34630849e-01, 1.02139493e-02, 3.14004149e-02, 2.29755094e-02],
                  [-5.80157882e-04, 2.15400039e-03, -1.68306824e-03, 2.71815458e-03,
                   4.20401479e-04, -2.78040546e-04, -1.36702638e-04, -1.16004163e-08,
                   1.91853582e-03, -8.42843245e-04, -1.09890912e-03, -2.79179570e-03],
                  [1.30766038e-01, -1.68306824e-03, 2.27442409e-01, 1.58248495e-03,
                   -3.01962404e-03, -2.22971118e-02, 6.01195487e-03, 2.70301593e-06,
                   -1.35577657e-01, 2.44200027e-02, 1.27094320e-02, 4.75666823e-02],
                  [-1.64303794e-02, 2.71815458e-03, 1.58248495e-03, 1.45371324e-01,
                   6.39877361e-03, 1.29483843e-02, 2.12866658e-02, -1.89630555e-06,
                   2.02699679e-02, 2.33181842e-02, 6.55835032e-03, -1.38957088e-02],
                  [-6.26605536e-03, 4.20401479e-04, -3.01962404e-03, 6.39877361e-03,
                   2.66722371e-02, 1.33016705e-02, 1.14185189e-02, -8.27503437e-07,
                   -2.55609076e-03, 2.21617390e-02, -3.50034455e-03, -9.32089590e-04],
                  [-2.60816538e-02, -2.78040546e-04, -2.22971118e-02, 1.29483843e-02,
                   1.33016705e-02, 2.75334961e-01, 1.71640169e-01, -4.34797550e-06,
                   -3.21135205e-04, 5.50358278e-02, -1.67370803e-02, -2.10103462e-02],
                  [-2.65245856e-02, -1.36702638e-04, 6.01195487e-03, 2.12866658e-02,
                   1.14185189e-02, 1.71640169e-01, 1.88606897e-01, -7.35735322e-06,
                   -2.10704376e-02, 3.55236581e-02, -2.13388953e-02, -6.44806344e-03],
                  [8.37288402e-06, -1.16004163e-08, 2.70301593e-06, -1.89630555e-06,
                   -8.27503437e-07, -4.34797550e-06, -7.35735322e-06, 8.49990939e-09,
                   -5.63006753e-06, 2.17821113e-06, 4.96260951e-06, -1.25886170e-06],
                  [-1.34630849e-01, 1.91853582e-03, -1.35577657e-01, 2.02699679e-02,
                   -2.55609076e-03, -3.21135205e-04, -2.10704376e-02, -5.63006753e-06,
                   2.34138446e-01, -4.37360972e-02, 3.39491727e-03, -4.13361145e-03],
                  [1.02139493e-02, -8.42843245e-04, 2.44200027e-02, 2.33181842e-02,
                   2.21617390e-02, 5.50358278e-02, 3.55236581e-02, 2.17821113e-06,
                   -4.37360972e-02, 2.14433967e-01, 6.32376986e-03, 1.21498681e-02],
                  [3.14004149e-02, -1.09890912e-03, 1.27094320e-02, 6.55835032e-03,
                   -3.50034455e-03, -1.67370803e-02, -2.13388953e-02, 4.96260951e-06,
                   3.39491727e-03, 6.32376986e-03, 9.00001321e-02, 5.84061237e-02],
                  [2.29755094e-02, -2.79179570e-03, 4.75666823e-02, -1.38957088e-02,
                   -9.32089590e-04, -2.10103462e-02, -6.44806344e-03, -1.25886170e-06,
                   -4.13361145e-03, 1.21498681e-02, 5.84061237e-02, 3.73502102e-01]])

improved_imputed_values = []
imputed_values = []
restored_data = []
median_impute = []
for test_data in damaged_data:
    missing_dim = [idx for idx, value in enumerate(test_data) if np.isnan(value)]

    imputed_value = nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)
    improved_imputed_value = improved_nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)

    restored_element = np.insert(test_data, missing_dim, imputed_value)
    # restored_data.append(restored_element)
    median_impute.append(medians[missing_dim])

    imputed_values.append(imputed_value)
    improved_imputed_values.append(improved_imputed_value)

median_impute = np.array(median_impute)
# restored_data = np.array(restored_data)
imputed_values = np.array(imputed_values)
improved_imputed_values = np.array(improved_imputed_values)

divergence = np.abs(np.array([np.mean(diff) for diff in (removed_values - imputed_values) / removed_values]) * 100)
improved_divergence = np.abs(
    np.array([np.mean(diff) for diff in (removed_values - improved_imputed_values) / removed_values]) * 100)
divergence_median = np.abs(np.array(
    [np.mean(diff) for diff in (removed_values - median_impute) / removed_values]) * 100)

mse = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values) ** 2])
improved_mse = np.array([np.mean(diff) for diff in np.abs(removed_values - improved_imputed_values) ** 2])
mse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute) ** 2])

plt.figure(2)
plt.plot(np.arange(len(divergence)), divergence, '-b', label='Error')
plt.plot(np.arange(len(improved_divergence)), improved_divergence, '-g', label='Improved Error')
plt.plot(np.arange(len(divergence_median)), divergence_median, '-r', label='Error median')
leg = plt.legend()
plt.xlabel('Index')
plt.ylabel('Imputation error %')
plt.show()

plt.figure(3)
plt.plot(np.arange(len(mse)), mse, '-b', label='MSE')
plt.plot(np.arange(len(improved_mse)), improved_mse, '-g', label='Improved MSE')
plt.plot(np.arange(len(mse_median)), mse_median, '-r', label='MSE median')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Imputation error MSE')
plt.show()

print(f'Median imputation error: {round(mse_median.mean(), 2)}')
print(f'Nadaraya-Watson imputation error: {round(mse.mean(), 2)}')
print(f'Improved Nadaraya-Watson imputation error: {round(improved_mse.mean(), 2)}')

print(f'Median imputation error (MSE): {round(mse_median.mean(), 2)}')
print(f'Nadaraya-Watson imputation error (MSE): {round(mse.mean(), 2)}')
print(f'Improved Nadaraya-Watson imputation error (MSE): {round(improved_mse.mean(), 2)}')
