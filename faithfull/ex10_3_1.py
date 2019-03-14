# exercise 10.1.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.cluster import k_means
from sklearn.preprocessing import scale

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/faithful.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, M = X.shape

X = scale(X, with_mean=True, with_std=True)

# Number of clusters:
K = 2

# K-means clustering:
centroids, cls, inertia = k_means(X,K,n_init=1)
    
# Plot results:
figure(figsize=(14,9))
clusterplot(X, cls, centroids, y)
show()

print('Ran Exercise 10.1.1')