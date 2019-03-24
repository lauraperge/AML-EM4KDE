from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import model_selection

## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
    l, V = np.linalg.eigh(Sigma)
    l[l < 0] = 0
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    xy = np.stack((np.cos(t), np.sin(t)))
    Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
    plt.plot(Txy[:, 0], Txy[:, 1])

# Load Matlab data file and extract variables of interest
mat_data = loadmat('./faithful.mat')
X = mat_data['X']
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, D = X.shape

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

max_iter = 30
Sigma_CVs = []

k = 0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    X_test = X[test_index, :]

    # initialize EM algorithm
    train_size = len(X_train)
    test_size = len(X_test)
    # 1 x X_train, each element is 2x2 matrix
    Sigma = [np.eye(D) for _ in range(test_size)]
    mixing_coeff = (1.0 / train_size) * np.ones(train_size)  # 1 x X_train
    responsibility = np.zeros(
        shape=(train_size, test_size))  # X_test x X_train
    log_likelihood = np.zeros(max_iter)  # 1 x iteration

    for iteration in range(max_iter):

        # E step
        for j, train in enumerate(X_train):
            resp_i = np.zeros(test_size)
            for i, test in enumerate(X_test):
                resp_i[i] = multivariate_normal.pdf(
                    test, mean=train, cov=Sigma[i])
            resp_i /= np.sum(resp_i, axis=0)
            responsibility[j, :] = resp_i[:]

        # M step
        sigma_test = []  # X_train sized list, each element is DxD
        for i, test in enumerate(X_test):
            sum_sigma_train = np.zeros(shape=(D, D))
            for j, train in enumerate(X_train):
                delta = test - train
                delta = np.matrix(delta)
                sum_sigma_train += (responsibility[j, i] * delta.T).dot(delta)
            sigma_test.append(sum_sigma_train)

        # Avg of Sigma created for each train point
        sum_sigma_test = np.zeros(shape=(D, D))
        for i, test in enumerate(X_test):
            sum_sigma_test += sigma_test[i]
        avg_sigma_test = sum_sigma_test / len(X_test)  # D x D

        # Update sigmas
        for i, test in enumerate(X_test):
            Sigma[i] = avg_sigma_test

        # calculate loglikelihood
        L = 0
        for i, test in enumerate(X_test):
            L_sub = 0
            for j, train in enumerate(X_train):
                L_sub += mixing_coeff[j] * \
                    multivariate_normal.pdf(
                        test, mean=train, cov=avg_sigma_test)
            L += np.log(L_sub)
        log_likelihood[iteration] = L
    # print(log_likelihood)
    Sigma_CVs.append(Sigma[0])

    k += 1

# the result of each CV fold stored in a list, it's a DxD covariance matrix obtained by performing EM algorithm 10 times
print(Sigma_CVs)

result_sigma = np.zeros(shape=(D, D))
for cv_fold in range(K):
    result_sigma += Sigma_CVs[cv_fold]
result_sigma /= K  # D x D

# the avg of the resulted Sigmas from CV folds
print(result_sigma)

## Plot data
plt.figure(2)
plt.plot(X[:, 0], X[:, 1], '.')
for _data in X:
    # print(_data)
    # print(sigma)
    plot_normal(_data, result_sigma)
plt.show()
