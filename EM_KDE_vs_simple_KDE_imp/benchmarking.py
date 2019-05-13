import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
from utils import remove_random_value, nadaraya_watson_imputation
from knn_imputation import knn_impute, find_null
import matplotlib.pyplot as plt

# Sigma for F-Kernel
sigma_F = np.array([[6.48119152e-04, 1.97261524e-05, 3.28995716e-05, 1.75467328e-04,
                     5.53304683e-06, -3.72432718e-04, -4.21233922e-04, 4.06141951e-05,
                     1.14670648e-04, 1.89826881e-05, 3.52954365e-04, 2.22300834e-04],
                    [1.97261524e-05, 8.16157072e-06, -1.54762417e-06, 1.27332809e-05,
                     5.07504989e-07, -7.58816603e-07, -2.77414572e-05, 3.46213656e-06,
                     1.20344595e-05, 1.45252595e-06, 3.26173134e-05, 6.50215187e-06],
                    [3.28995716e-05, -1.54762417e-06, 6.11577717e-06, 1.31000627e-05,
                     1.91241843e-07, -3.35425225e-05, -1.47965279e-05, 7.34329848e-07,
                     5.90729211e-07, 1.09060628e-06, 6.54676389e-06, 8.76059867e-06],
                    [1.75467328e-04, 1.27332809e-05, 1.31000627e-05, 2.44634013e-04,
                     3.81217214e-06, -1.56352584e-04, -2.47345607e-04, 1.50084065e-05,
                     4.84157381e-05, 1.03766198e-05, 2.20036688e-04, 7.65131728e-05],
                    [5.53304683e-06, 5.07504989e-07, 1.91241843e-07, 3.81217214e-06,
                     2.33206129e-07, -3.28466371e-06, -7.83399152e-06, 7.00015419e-07,
                     2.33877308e-06, 6.89912373e-07, 7.51433018e-06, 3.87134436e-06],
                    [-3.72432718e-04, -7.58816603e-07, -3.35425225e-05, -1.56352584e-04,
                     -3.28466371e-06, 1.91523995e-03, -4.10475438e-04, -7.68195651e-06,
                     -1.02589085e-05, -4.11535493e-06, -1.15427325e-04, -6.03481937e-05],
                    [-4.21233922e-04, -2.77414572e-05, -1.47965279e-05, -2.47345607e-04,
                     -7.83399152e-06, -4.10475438e-04, 8.97538715e-04, -5.38355036e-05,
                     -1.79734479e-04, -3.28509088e-05, -5.90570324e-04, -3.38682201e-04],
                    [4.06141951e-05, 3.46213656e-06, 7.34329848e-07, 1.50084065e-05,
                     7.00015419e-07, -7.68195651e-06, -5.38355036e-05, 5.94088578e-06,
                     2.01009921e-05, 3.41455695e-06, 5.53657959e-05, 2.72525962e-05],
                    [1.14670648e-04, 1.20344595e-05, 5.90729211e-07, 4.84157381e-05,
                     2.33877308e-06, -1.02589085e-05, -1.79734479e-04, 2.01009921e-05,
                     7.18571328e-05, 1.15095333e-05, 1.92999232e-04, 9.17542248e-05],
                    [1.89826881e-05, 1.45252595e-06, 1.09060628e-06, 1.03766198e-05,
                     6.89912373e-07, -4.11535493e-06, -3.28509088e-05, 3.41455695e-06,
                     1.15095333e-05, 5.93219807e-06, 3.55081410e-05, 2.34873171e-05],
                    [3.52954365e-04, 3.26173134e-05, 6.54676389e-06, 2.20036688e-04,
                     7.51433018e-06, -1.15427325e-04, -5.90570324e-04, 5.53657959e-05,
                     1.92999232e-04, 3.55081410e-05, 6.68918465e-04, 3.09595531e-04],
                    [2.22300834e-04, 6.50215187e-06, 8.76059867e-06, 7.65131728e-05,
                     3.87134436e-06, -6.03481937e-05, -3.38682201e-04, 2.72525962e-05,
                     9.17542248e-05, 2.34873171e-05, 3.09595531e-04, 4.08194717e-04]])

# Sigma for D-Kernel
sigma_D = np.array([[4.66784973e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 7.13859815e-06, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 4.61235791e-06, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.68764666e-04,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     3.33381214e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 8.08115189e-04, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 2.19661809e-04, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.39194558e-06,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     1.88260155e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 7.21498027e-06, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 2.03868890e-04, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.04664093e-04]])

# Sigma for S-Kernel
sigma_S = np.array([[0.32581447, 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0.32581447, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0.32581447, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0.32581447, 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.32581447, 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.32581447,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0.32581447, 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0.32581447, 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0.32581447, 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0.32581447, 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.32581447, 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.32581447]])

# Load data
raw_data = preprocessing.scale(loadmat('../faithfull/wine.mat')['X'])

# Remove attributes randomly
NUM_TEST = 100
raw_data = raw_data[:1000]  # taking only a small part for testing
data = np.array(raw_data[:-NUM_TEST])
[damaged_data, removed_values] = remove_random_value(raw_data[-NUM_TEST:])
medians = np.median(data, axis=0)  # for baseline

# Reformat removed_values for the KNN imputation
flattened_removed_values = [y for x in removed_values for y in x]

# Information
print('{0} attributes were removed from the first {1} data point.'.format(len(flattened_removed_values), NUM_TEST))

# Storage for benchmarking results
imputed_values = []
median_impute = []
benchmarks = [sigma_F, sigma_D, sigma_S]
do_median = True

# Benchmark the performance of EM_KDE with different kernels
print('Benchmarking different kernels...')
for benchmark_case in benchmarks:
    sigma = benchmark_case
    _imputed_values = []
    for test_data in damaged_data:
        imputed_value = nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)
        _imputed_values.append(imputed_value)

        if do_median:
            missing_dim = [idx for idx, value in enumerate(test_data) if np.isnan(value)]
            median_impute.append(medians[missing_dim])

    do_median = False
    _imputed_values = np.array(_imputed_values)
    imputed_values.append(_imputed_values)

median_impute = np.array(median_impute)
imputed_values = np.array(imputed_values)

# Benchmark imputation with KNN method
print('Benchmarking KNN imputation...')
damaged_data_set = np.append(damaged_data, data, axis=0)
neighbors = np.arange(1, 109, 2)
avg_mse = []
best_neighbor_avg_mse = 100
best_neighbor_number = 0
best_neighbor_mse = []
best_neighbor_rmse = []
for neighbor in neighbors:
    imputed_data = knn_impute(damaged_data_set, k=neighbor)
    imputed_values_knn = np.array([imputed_data.item(tuple(x)) for x in find_null(damaged_data)])
    mse_knn = np.array(np.abs(flattened_removed_values - imputed_values_knn) ** 2)
    rmse_knn = np.array(np.abs(flattened_removed_values - imputed_values_knn))
    avg_mse.append(np.average(mse_knn))
    if np.average(mse_knn) < best_neighbor_avg_mse:
        best_neighbor_avg_mse = np.average(mse_knn)
        best_neighbor_mse = mse_knn
        best_neighbor_rmse = rmse_knn
        best_neighbor_number = neighbor

print('The best number of neighbor to use is {0}'.format(best_neighbor_number))

# Results
plt.figure(2)
plt.plot(neighbors, avg_mse)
plt.xlabel('Number of neighbors')
plt.ylabel('Average MSE error')
plt.show()

mse_F = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[0]) ** 2])
mse_D = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[1]) ** 2])
mse_S = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[2]) ** 2])
mse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute) ** 2])

rmse_F = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[0])])
rmse_D = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[1])])
rmse_S = np.array([np.mean(diff) for diff in np.abs(removed_values - imputed_values[2])])
rmse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute)])

plt.figure(3)
plt.plot(np.arange(len(mse_S)), mse_S, '-y', label='MSE (S-Kernel)')
plt.plot(np.arange(len(mse_D)), mse_D, '-r', label='MSE (D-Kernel)')
plt.plot(np.arange(len(mse_F)), mse_F, '-b', label='MSE (F-Kernel)')
# plt.plot(np.arange(len(mse_median)), mse_median, '-r', label='MSE median')
# plt.plot(np.arange(len(best_neighbor_mse)), best_neighbor_mse, '-m', label='MSE KNN ({0})'.format(best_neighbor_number))
plt.legend()
plt.xlabel('Index of damaged observation')
plt.ylabel('Imputation MSE')
plt.show()

plt.figure(4)
plt.plot(np.arange(len(rmse_S)), rmse_S, '-y', label='RMSE (S-Kernel)')
plt.plot(np.arange(len(rmse_D)), rmse_D, '-r', label='RMSE (D-Kernel)')
plt.plot(np.arange(len(rmse_F)), rmse_F, '-b', label='RMSE (F-Kernel)')
# plt.plot(np.arange(len(rmse_median)), rmse_median, '-r', label='RMSE median')
# plt.plot(np.arange(len(best_neighbor_rmse)), best_neighbor_rmse, '-m', label='MSE KNN ({0})'.format(best_neighbor_number))
plt.legend()
plt.xlabel('Index of damaged observation')
plt.ylabel('Imputation RMSE')
plt.show()


boxplot_data = [mse_F, mse_D, mse_S, mse_median, best_neighbor_mse]
plt.figure(5)
plt.title('Imputation MSE')
plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
plt.show()

boxplot_data = [rmse_F, rmse_D, rmse_S, rmse_median, best_neighbor_rmse]
plt.figure(6)
plt.title('Imputation RMSE')
plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
plt.show()