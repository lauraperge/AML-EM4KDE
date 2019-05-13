import numpy as np
from scipy.spatial import KDTree
from scipy.io import loadmat
from utils import remove_random_value
import matplotlib.pyplot as plt
from sklearn import preprocessing


def find_null(data):
  """ Finds the indices of all missing values.
    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.
    Returns
    -------
    List of tuples
        Indices of all missing values in tuple format; (i, j)
  """

  nulls = np.argwhere(np.isnan(data))
  return nulls

def fill_mean(data) :
  """ Substitute missing values with the mean of that column.
    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.
    Returns
    -------
    numpy.ndarray
        Imputed data.
  """

  # Finds the indices of all missing values.
  nulls = find_null(data)
  # Returns np.matrix, first index is thw index of the datapoint missing an attribute
  # Second index is the dimension which is missing
  # [[3,1][4,1][5,0]]
  for datapoint_i, missing_dim_i in nulls:
    # Get all the non-nan attributes in that particular dimension
    row_wo_nan = data[:, [missing_dim_i]][~np.isnan(data[:, [missing_dim_i]])]
    # Calculate the mean of that attribute
    new_value = np.mean(row_wo_nan)
    # Set the NaN value to be the mean
    data[datapoint_i][missing_dim_i] = new_value
  return data


def knn_impute(data, k=3):
  """ Impute missing data with a nearest neighbor approach

  Impute missing attributes with mean imputation and then use the resulting complete
  dataset to construct a KDTree. Use this KDTree to compute nearest neighbors.
  After finding `k` nearest neighbors, take the weighted average of their attribute.
   
  Parameters
    ----------
    data: numpy.ndarray
        Data to impute.
    k: int
        number of neighbors
    Returns
    -------
    numpy.ndarray
        Imputed data.
  """
  imputed_data = np.copy(data)
  nulls = find_null(imputed_data)
  data_corrected = fill_mean(imputed_data)
  # Use all the default parameters, calculate based on Euclidean distance
  kdTree = KDTree(data_corrected)

  for datapoint_i, missing_dim_i in nulls:
    distances, indices = kdTree.query(data_corrected[datapoint_i], k=k+1)
    # Will always return itself in the first index. Delete it.
    distances, indices = distances[1:], indices[1:]
    if np.sum(distances) != 0:
        weights = distances/np.sum(distances)
    else:
        weights = distances
    # Assign the weighted average of `k` nearest neighbors
    imputed_data[datapoint_i][missing_dim_i] = np.dot(weights, [data_corrected[ind][missing_dim_i] for ind in indices])

  return imputed_data


if __name__ == '__main__':
    ## Load data
    full_data = loadmat('../faithfull/wine.mat')['X']
    full_data = preprocessing.normalize(full_data)

    full_part = np.array(full_data[10:])
    full_data_copy = np.copy(full_data)

    [damaged_part, removed_values] = remove_random_values(full_data[:10])

    damaged_data = np.append(damaged_part, full_part, axis=0)

    neighbors = np.arange(1, 19, 2)
    avg_divergence = []

    for neighbor in neighbors:
        imputed_data = knn_impute(damaged_data, k=neighbor)
        imputed_part = np.array(imputed_data[:10])
        imputed_values = np.array([imputed_data.item(tuple(x)) for x in find_null(damaged_part)])
        divergence = []
        for i in range(len(removed_values)):
            if removed_values[i] != 0:
                divergence.append(np.abs(removed_values[i] - imputed_values[i]) / removed_values[i])
            else:
                divergence.append(np.abs(removed_values[i] - imputed_values[i]) / imputed_values[i])
        avg_divergence.append(np.average(divergence))

    plt.figure(2)
    plt.plot(neighbors, avg_divergence)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Average imputation error')
    plt.show()
