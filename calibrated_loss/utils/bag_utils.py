import numpy as np

def make_bags(X, y, N):
    #transform the feature vector and the label vector, y, into bags of 
    #features with label proportion.
    #Input:
    #  X: feature vector, of shape (num_samples, dim_features)
    #  y: label vector, of shape (num_samples, )
    #  N: bag size, must divide X.shape[0]
    #Return: 
    #  X_bags: a python list of numpy arrays with shape (N, dim_features);
    #  z0: a numpy array with shape (num_samples/N, ), z[i] contains the 
    #      proportion of 1's in ith bag
    assert(X.shape[0] == y.shape[0])
    assert(X.shape[0]%(2*N) == 0)
    num_bags = X.shape[0]//N
    X_bags = []
    z = np.zeros((num_bags, ))
    for i in range(num_bags):
        X_bags.append(X[i*N:(i+1)*N, :].copy())
        z[i] = np.sum(y[i*N:(i+1)*N] == 1)/N
    return X_bags, z
  
def compute_alpha(z0, z1):
    #compute the parameter alpha
    assert(z1 >= z0)
    if z1 + z0 == 0:
        p1_ = 0.0
        p0_ = 0.5
    elif z1 + z0 == 2:
        p1_ = 0.5
        p0_ = 0.0
    else: 
        p1_ = z0/(z0 + z1)
        p0_ = (1 - z1)/(2 - z0 - z1)
    alpha = 0.5 + 0.5 * (p0_ - p1_)
    return alpha
  
def recover(X_bags, z, pairs, shuffle=True):
    #calculate the estimated label and calibration factor alpha for each 
    #instance
    #Input:
    #  X_bags: a python list of numpy arrays with shape (N, dim_features)
    #  z0: a numpy array with shape (num_samples/N, ), z[i] contains the 
    #      proportion of 1's in ith bag
    #Return:
    #  X: the feature vector
    #  y: the estimated label
    #  pairs: a list of tuples of index pairs representing the pairing of bags
    N, d = X_bags[0].shape
    M = 2 * len(pairs)
    X = np.zeros((N*M, d))
    y = np.zeros((N*M, ))
    w = np.zeros((N*M, ))
    m = 0#count the number of processed bag
    for pair in pairs:
        i, j = pair
        if z[i] > z[j]:
            temp = i
            i = j
            j = temp
        z0, z1 = z[i], z[j]
        bag0, bag1 = X_bags[i], X_bags[j]
        alpha = compute_alpha(z0, z1)
        X[m*N:(m+1)*N, :] = bag0
        y[m*N:(m+1)*N] = 0
        w[m*N:(m+1)*N] = alpha
        m += 1
        X[m*N:(m+1)*N, :] = bag1
        y[m*N:(m+1)*N] = 1
        w[m*N:(m+1)*N] = alpha
        m += 1
    if shuffle:
        p = np.random.permutation(N*M)
        X = X[p]
        y = y[p]
        w = w[p]
    return X, y, w

def pair_to_1(z):
    #pair the bags s.t. each z0, z1 sums to 1, approximately
    #Input: a list of z values
    #Returns: a list of tuples, each tuple is a pair of indices corresponding to
    #         the position in the list z
    res = []
    chosen = set([])
    for i in range(len(z) - 1):
        if len(chosen) - len(z) == -1:
            break
        if i in chosen:
            continue
        z0 = z[i]
        min_val = 1
        min_index = -1
        for j in range(i+1, len(z)):
            comp = True
            if j in chosen:
                continue
            z1 = z[j]
            if z0 + z1 == 1:
                res.append((i, j))
                chosen.add(i)
                chosen.add(j)
                comp = False
                break
            comp = True
            temp = abs(z0 + z1 - 1)
            if temp <= min_val:
                min_val = temp
                min_index = j
        if comp:
            assert(min_index != -1)
            res.append((i, min_index))
            chosen.add(i)
            chosen.add(min_index)
    return res

def compute_weights(alpha, function):
    #compute the weight assigned to each training example based on alpha
    #Input: alpha, the alpha value of each training instance
    #       function, a function maps [0, 1] to a positive real number;
    #                 it maps |p0 - p1| to a corresponding weight
    diff = np.abs(2*alpha-1)
    return function(diff)

def neg_exp(a):
    func = lambda t: np.exp(-a*t)
    return func