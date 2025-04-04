import numpy as np
import pickle
import os


def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte