from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import scale
from sklearn import model_selection

# Load Matlab data file and extract variables of interest
mat_data = loadmat('./faithful.mat')
X = mat_data['X']
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, D = X.shape

X_scaled = scale(X, with_mean=True, with_std=True)

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test = model_selection.train_test_split(X,test_size=test_proportion)
