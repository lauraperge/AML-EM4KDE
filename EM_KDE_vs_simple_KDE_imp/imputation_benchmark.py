import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
from EM_KDE_vs_simple_KDE_imp.utils import remove_random_value, nadaraya_watson_imputation
from EM_KDE_vs_simple_KDE_imp.knn_imputation import knn_impute, find_null
import matplotlib.pyplot as plt

# Sigma for F-Kernel
sigma_F = np.array([[1.90367147e-01, -5.54762511e-04, 1.34608492e-01, -1.44144445e-02,
                     -6.46673923e-03, -2.80929912e-02, -2.62877588e-02, 8.87417370e-06,
                     -1.38812455e-01, 1.01308815e-02, 3.01821224e-02, 2.14599690e-02],
                    [-5.54762511e-04, 1.91730674e-03, -1.57539240e-03, 2.44562469e-03,
                     3.98907323e-04, -2.56091062e-04, -1.25519855e-04, -1.10521083e-08,
                     1.80983780e-03, -8.05834154e-04, -9.97230715e-04, -2.55432205e-03],
                    [1.34608492e-01, -1.57539240e-03, 2.26223757e-01, 3.32107739e-03,
                     -2.59828445e-03, -2.01023060e-02, 7.53425627e-03, 2.67647332e-06,
                     -1.38282061e-01, 2.40416860e-02, 1.21056160e-02, 4.68469731e-02],
                    [-1.44144445e-02, 2.44562469e-03, 3.32107739e-03, 1.34454943e-01,
                     7.31525279e-03, 1.26318135e-02, 2.08659707e-02, -1.92227352e-06,
                     1.86259567e-02, 2.31518856e-02, 6.13567215e-03, -1.29881746e-02],
                    [-6.46673923e-03, 3.98907323e-04, -2.59828445e-03, 7.31525279e-03,
                     2.78504534e-02, 1.41431647e-02, 1.19578145e-02, -8.45160342e-07,
                     -1.38877960e-03, 2.34979998e-02, -3.07714641e-03, -7.23556335e-04],
                    [-2.80929912e-02, -2.56091062e-04, -2.01023060e-02, 1.26318135e-02,
                     1.41431647e-02, 2.74777917e-01, 1.65786689e-01, -4.45933920e-06,
                     2.24394733e-03, 5.66060944e-02, -1.39813620e-02, -2.23238134e-02],
                    [-2.62877588e-02, -1.25519855e-04, 7.53425627e-03, 2.08659707e-02,
                     1.19578145e-02, 1.65786689e-01, 1.76208386e-01, -7.40683886e-06,
                     -1.98539869e-02, 3.49157188e-02, -1.90580234e-02, -6.81570999e-03],
                    [8.87417370e-06, -1.10521083e-08, 2.67647332e-06, -1.92227352e-06,
                     -8.45160342e-07, -4.45933920e-06, -7.40683886e-06, 9.14245330e-09,
                     -5.89691351e-06, 2.54202256e-06, 5.06828447e-06, -1.23260156e-06],
                    [-1.38812455e-01, 1.80983780e-03, -1.38282061e-01, 1.86259567e-02,
                     -1.38877960e-03, 2.24394733e-03, -1.98539869e-02, -5.89691351e-06,
                     2.36984659e-01, -4.04945287e-02, 5.33263621e-03, -2.14470426e-03],
                    [1.01308815e-02, -8.05834154e-04, 2.40416860e-02, 2.31518856e-02,
                     2.34979998e-02, 5.66060944e-02, 3.49157188e-02, 2.54202256e-06,
                     -4.04945287e-02, 2.25222724e-01, 9.15702868e-03, 1.20695298e-02],
                    [3.01821224e-02, -9.97230715e-04, 1.21056160e-02, 6.13567215e-03,
                     -3.07714641e-03, -1.39813620e-02, -1.90580234e-02, 5.06828447e-06,
                     5.33263621e-03, 9.15702868e-03, 8.49887939e-02, 5.32675371e-02],
                    [2.14599690e-02, -2.55432205e-03, 4.68469731e-02, -1.29881746e-02,
                     -7.23556335e-04, -2.23238134e-02, -6.81570999e-03, -1.23260156e-06,
                     -2.14470426e-03, 1.20695298e-02, 5.32675371e-02, 3.35970796e-01]])

# Sigma for D-Kernel
sigma_D = np.array([[1.31914474e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 1.91763074e-03, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 1.70848327e-01, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.39660780e-01,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     2.59918563e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 1.77133673e-01, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 1.14018984e-01, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.26289018e-08,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     1.70726786e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 2.58829611e-01, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 1.13052627e-01, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.30431250e-01]])

# Sigma for S-Kernel
sigma_S = np.array([[0.10264579, 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0.10264579, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0.10264579, 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0.10264579, 0., 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.10264579, 0.,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.10264579,
                     0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0.10264579, 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0.10264579, 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0.10264579, 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0.10264579, 0., 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.10264579, 0.],
                    [0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0.10264579]])

# Load data
# Taking only a small part for testing
raw_data = preprocessing.scale(loadmat('../faithfull/wine.mat')['X'][:1000])

# Remove attributes randomly
NUM_TEST = 100
data = np.array(raw_data[:-NUM_TEST])
[damaged_data, removed_values] = remove_random_value(raw_data[-NUM_TEST:])
medians = np.median(data, axis=0)  # for baseline

# Reformat removed_values for the KNN imputation
flattened_removed_values = [y for x in removed_values for y in x]

# Information
print('{0} attributes were removed from the first {1} data point.'.format(len(flattened_removed_values), NUM_TEST))

# Storage for benchmarking results
imputed_values = []
cond_exp_imputed_values = []
median_impute = []
benchmarks = [sigma_F, sigma_D, sigma_S]
do_median = True

# Benchmark the performance of EM_KDE with different kernels
print('Benchmarking different kernels...')
for benchmark_case in benchmarks:
    sigma = benchmark_case
    _imputed_values = []
    _cond_exp_imputed_values = []
    for test_data in damaged_data:
        imputed_value = nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)
        cond_exp_imputed_value = improved_nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma)
        _imputed_values.append(imputed_value)
        _cond_exp_imputed_values.append(cond_exp_imputed_value)

        if do_median:
            missing_dim = [idx for idx, value in enumerate(test_data) if np.isnan(value)]
            median_impute.append(medians[missing_dim])

    do_median = False
    _imputed_values = np.array(_imputed_values)
    imputed_values.append(_imputed_values)
    _cond_exp_imputed_values = np.array(_cond_exp_imputed_values)
    cond_exp_imputed_values.append(_cond_exp_imputed_values)

median_impute = np.array(median_impute)
imputed_values = np.array(imputed_values)
cond_exp_imputed_values = np.array(cond_exp_imputed_values)

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

cond_exp_mse_F = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[0]) ** 2])
cond_exp_mse_D = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[1]) ** 2])
cond_exp_mse_S = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[2]) ** 2])
cond_exp_mse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute) ** 2])

cond_exp_rmse_F = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[0])])
cond_exp_rmse_D = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[1])])
cond_exp_rmse_S = np.array([np.mean(diff) for diff in np.abs(removed_values - cond_exp_imputed_values[2])])
cond_exp_rmse_median = np.array([np.mean(diff) for diff in np.abs(removed_values - median_impute)])


plt.figure(3)
plt.plot(np.arange(len(mse_S)), mse_S, '-y', label='MSE (S-Kernel)')
plt.plot(np.arange(len(mse_D)), mse_D, '-r', label='MSE (D-Kernel)')
plt.plot(np.arange(len(mse_F)), mse_F, '-b', label='MSE (F-Kernel)')
plt.legend()
plt.xlabel('Index of damaged observation')
plt.ylabel('Imputation MSE')
plt.show()

plt.figure(4)
plt.plot(np.arange(len(rmse_S)), rmse_S, '-y', label='RMSE (S-Kernel)')
plt.plot(np.arange(len(rmse_D)), rmse_D, '-r', label='RMSE (D-Kernel)')
plt.plot(np.arange(len(rmse_F)), rmse_F, '-b', label='RMSE (F-Kernel)')
plt.legend()
plt.xlabel('Index of damaged observation')
plt.ylabel('Imputation RMSE')
plt.show()


# boxplot_data = [mse_F, mse_D, mse_S, mse_median, best_neighbor_mse]
# plt.figure(5)
# plt.title('Imputation MSE')
# plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
# plt.show()
#
# boxplot_data = [rmse_F, rmse_D, rmse_S, rmse_median, best_neighbor_rmse]
# plt.figure(6)
# plt.title('Imputation RMSE')
# plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
# plt.show()

boxplot_data = [mse_F, mse_D, mse_S, cond_exp_mse_F, cond_exp_mse_D, cond_exp_mse_S, mse_median, best_neighbor_mse]
plt.figure(5)
plt.title('Imputation MSE')
plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
plt.show()

boxplot_data = [rmse_F, rmse_D, rmse_S, cond_exp_rmse_F, cond_exp_rmse_D, cond_exp_rmse_S, rmse_median, best_neighbor_rmse]
plt.figure(6)
plt.title('Imputation RMSE')
plt.boxplot(boxplot_data, showfliers=False, labels=['F-kernel', 'D-kernel', 'S-kernel', 'F-kernel', 'D-kernel', 'S-kernel', 'Median', 'KNN ({0})'.format(best_neighbor_number)], patch_artist=True)
plt.show()
