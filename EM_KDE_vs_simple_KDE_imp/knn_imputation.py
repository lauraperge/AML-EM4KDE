import numpy as np
from scipy.spatial import KDTree


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


def fill_mean(data):
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

    def get_kd_tree():
        imputed_data = np.copy(data)
        nulls = find_null(imputed_data)
        data_corrected = fill_mean(imputed_data)
        # Use all the default parameters, calculate based on Euclidean distance
        kdTree = KDTree(data_corrected)

        return [imputed_data, nulls, data_corrected, kdTree]

    [imputed_data, nulls, data_corrected, kdTree] = get_kd_tree()

    def get_immputation(imputed_data, nulls, data_corrected, kdTree):
        for datapoint_i, missing_dim_i in nulls:
            distances, indices = kdTree.query(data_corrected[datapoint_i], k=k + 1)
            # Will always return itself in the first index. Delete it.
            distances, indices = distances[1:], indices[1:]
            if np.sum(distances) != 0:
                weights = distances / np.sum(distances)
            else:
                weights = distances
            # Assign the weighted average of `k` nearest neighbors
            imputed_data[datapoint_i][missing_dim_i] = np.dot(weights,
                                                              [data_corrected[ind][missing_dim_i] for ind in indices])
        return imputed_data

    imputed_data = get_immputation(imputed_data, nulls, data_corrected, kdTree)

    return imputed_data
