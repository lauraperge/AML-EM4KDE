import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
from EM_KDE_vs_simple_KDE_imp.utils import remove_random_value, nadaraya_watson_imputation, \
    improved_nadaraya_watson_imputation

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
# Load data
# Taking only a small part for testing
raw_data = preprocessing.scale(loadmat('../faithfull/wine.mat')['X'][:1000])

# Remove attributes randomly
NUM_TEST = 100
data = np.array(raw_data[:-NUM_TEST])
[damaged_data, removed_values] = remove_random_value(raw_data[-NUM_TEST:])

# Reformat removed_values for the KNN imputation
flattened_removed_values = [y for x in removed_values for y in x]

# Information
print('{0} attributes were removed from the first {1} data point.'.format(len(flattened_removed_values), NUM_TEST))

# Storage for benchmarking results
imputed_values = []
cond_exp_imputed_values = []
median_impute = []


def get_median():
    return np.median(data, axis=0)


medians = get_median()  # for baseline


def impute():
    # Benchmark the performance of EM_KDE with different kernels
    print('Benchmarking different kernels...')

    _imputed_values = []
    _cond_exp_imputed_values = []

    for test_data in damaged_data:
        # imputed_value = nadaraya_watson_imputation(damaged_data=test_data, train_data=data, sigma=sigma_F)
        # _imputed_values.append(imputed_value)

        # cond_exp_imputed_value = improved_nadaraya_watson_imputation(damaged_data=test_data, train_data=data,
        #                                                              sigma=sigma_F)
        # _cond_exp_imputed_values.append(cond_exp_imputed_value)

        missing_dim = [idx for idx, value in enumerate(test_data) if np.isnan(value)]
        median_impute.append(medians[missing_dim])


impute()
