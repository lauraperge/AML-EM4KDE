import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def plot_kde(data, sigma, res):
    [min_x, min_y] = np.min(data, axis=0)
    [max_x, max_y] = np.max(data, axis=0)
    border_x = 0.2 * (max_x - min_x)
    border_y = 0.2 * (max_y - min_y)
    x_range = np.arange(min_x - border_x, max_x + border_x, res)
    y_range = np.arange(min_y - border_y, max_y + border_y, res)
    z = evaluate_probability(x_range, y_range, data, sigma)
    h = plt.contourf(x_range, y_range, z)
    plt.colorbar()
    plt.show()


def evaluate_probability(x_range, y_range, data, sigma):
    x_size = len(x_range)
    y_size = len(y_range)
    probability = np.zeros((y_size, x_size))
    for i in range(x_size):
        for j in range(y_size):
            test_point = np.array([x_range[i], y_range[j]])
            probabilities = multivariate_normal.pdf(data, mean=test_point, cov=sigma)
            probability[j][i] = probabilities.sum()
    return probability


if __name__ == '__main__':
    ## JUST TESTING
    data = np.array([
        [-3, -4],
        [1, 1],
        [-2, 1],
        [2, -2],
        [-3, 0]
        # [0,0],
        # [0,1],
        # [1,0],
        # [0,-1],
        # [-1,0],
        # [3,3]
    ])
    sigma = np.array([[2, -0.3], [-0.3, 0.5]])

    plot_kde(data, sigma, 0.1)
