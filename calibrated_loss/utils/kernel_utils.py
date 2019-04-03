def gaussian_kernel(sigma=1):
    def gaussian_f(x1, x2):
        return np.exp(-((x1-x2)*(x1-x2)).sum()/(2*sigma**2))
    return gaussian_f

def add_one(X):
    #add an column of 1 in the first column
    N, d = X.shape
    X_ = np.zeros((N, d+1))
    X_[:, 0] = np.ones((N,))
    X_[:, 1:] = X
    return X_

class Weighted_Ridge_Regression:
    
    def __init__(self, kernel, weights, lambd):
        self.beta = None
        self.kernel = kernel
        self.weights = weight
        self.lambd = lambd

    def train(X, y):
        #output an classifier
        #variables:
        kernel = self.kernel
        weights = self.weights
        lambd = self.lambd
        N = y.shape[0]
        D = X.shape[1] + 1
        X_ = add_one(X)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = kernel(X_[i, :], X_[j, :])
        sqrt_w = np.sqrt(weights)
        sqrt_w_dig = np.zeros((N, N))
        for i in range(N):
            sqrt_w_dig[i, i] = sqrt_w[i]
        temp0 = np.matmul(sqrt_w.reshape((1, -1)), K)
        temp0 = np.matmul(temp0, sqrt_w.reshape((-1, 1)))
        temp0 = temp0 + lambd*np.eye(N)
        temp0 = np.linalg.inv(temp0)
        temp1 = np.multiply(sqrt_w, y).reshape((-1, 1))
        beta = np.matmul(np.matmul(sqrt_w_dig, temp0), temp1)
        self.beta = beta.reshape((-1,))

    def regress(X_train, X_test):
        kernel = self.kernel
        beta = self.beta
        if (type(beta) == type(None) ):
            raise Exception("Model is not trained")
        N, D = X_train.shape
        M = X_test.shape[0]
        K = np.zeros((M, N))
        X_train_ = add_one(X_train)
        X_test_ = add_one(X_test)
        for i in range(M):
            for j in range(N):
                K[i, j] = kernel(X_train_[j, :], X_test_[i, :])
        return np.matmul(K, beta)
