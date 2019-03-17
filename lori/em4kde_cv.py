from scipy.io import loadmat
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import model_selection

# Load Matlab data file and extract variables of interest
mat_data = loadmat('./faithful.mat')
X = mat_data['X']
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, D = X.shape

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

max_iter = 10
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
    Sigma = [np.eye(D) for _ in range(train_size)]
    mixing_coeff = (1.0 / train_size) * np.ones(train_size)  # 1 x X_train
    responsibility = np.zeros(
        shape=(test_size, train_size))  # X_test x X_train
    log_likelihood = np.zeros(max_iter)  # 1 x iteration

    for iteration in range(max_iter):

        # E step
        for i, test in enumerate(X_test):
            resp_i = np.zeros(train_size)
            for j, train in enumerate(X_train):
                resp_i[j] = multivariate_normal.pdf(
                    test, mean=train, cov=Sigma[j])
            resp_i /= np.sum(resp_i, axis=0)
            responsibility[i, :] = resp_i[:]

        # M step
        sigma_train = []  # X_train sized list, each element is DxD
        for j, train in enumerate(X_train):
            sum_sigma_test = np.zeros(shape=(D, D))
            for i, test in enumerate(X_test):
                delta = test - train
                delta = np.matrix(delta)
                sum_sigma_test += (responsibility[i, j] * delta.T).dot(delta)
            sigma_train.append(sum_sigma_test)

        # Avg of Sigma created for each train point
        sum_sigma_train = np.zeros(shape=(D, D))
        for j, train in enumerate(X_train):
            sum_sigma_train += sigma_train[j]
        avg_sigma_train = sum_sigma_train / len(X_train)  # D x D

        # Update sigmas
        for j, train in enumerate(X_train):
            Sigma[j] = avg_sigma_train

        # calculate loglikelihood
        L = 0
        for i, test in enumerate(X_test):
            L_sub = 0
            for j, train in enumerate(X_train):
                L_sub += mixing_coeff[j] * \
                    multivariate_normal.pdf(
                        test, mean=train, cov=Sigma[j])
            L += np.log(L_sub)
        log_likelihood[iteration] = L
    print(log_likelihood)
    Sigma_CVs.append(Sigma[0])

    k += 1

print(Sigma_CVs)