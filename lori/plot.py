import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np

def plot_kde(data, sigma, resolution):
  min_data = np.min(data, axis=0)
  # min_data = np.min(data, axis=1)
  max_data = np.max(data, axis=0)
  # min_data_0 = np.argmin(data)
  # min_data_1 = np.argmin(data, axis=1)
  print(min_data, max_data)
  x = np.arange(min_data[0], max_data[0], resolution)
  y = np.arange(min_data[1], max_data[1], resolution)
  xx, yy = np.meshgrid(x, y, sparse=True)
  z = evaluate_probability(xx, yy, data, sigma)
  h = plt.contourf(x, y, z)
  plt.show()

def evaluate_probability(x_range,y_range,data,sigma):
  x_size = x_range.shape[1]
  y_size = y_range.shape[0]
  print(x_size,y_size)
  probability = np.zeros((x_size,y_size))
  for i in range(x_size):
    for j in range(y_size):
      test_point = np.concatenate([x_range[:,i],y_range[j]])
      probabilities = multivariate_normal.pdf(data, mean=test_point, cov=sigma)
      probability[j][i] = probabilities.sum()
  return probability

data = np.array([
  [-3,-4],
  [2,2],
  [-2,5],
  [2,-2],
  [-4,0]
])
sigma = np.array([[0.5, 0], [0, 0.5]])

plot_kde(data, sigma, 1)

# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# xx, yy = np.meshgrid(x, y, sparse=True)
# z = evaluate_probability(xx,yy,data,sigma)
# h = plt.contourf(x,y,z)
# plt.show()
