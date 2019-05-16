import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat
from EM_KDE_vs_simple_KDE_imp.utils import remove_random_value
from EM_KDE_vs_simple_KDE_imp.knn_imputation import knn_impute, find_null
import matplotlib.pyplot as plt
from EM_KDE_vs_simple_KDE_imp.knn_imputation import fill_mean, KDTree

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

# Benchmark imputation with KNN method
print('Benchmarking KNN imputation...')
damaged_data_set = np.append(damaged_data, data, axis=0)


imputed_data = knn_impute(damaged_data_set, k=27)
