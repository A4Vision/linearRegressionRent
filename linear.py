import time

import numpy as np
from sklearn import linear_model


def sk_linear_regeression(X, y):
    r = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    r.fit(X, y)
    return r.coef_, r.intercept_


def residuals(X, y, coef, intercept):
    return np.dot(X, coef) + intercept - y


def wrap_linear_regression_solver(X, y, solver):
    extendedX = np.column_stack((X, np.ones((X.shape[0], 1))))
    coefs = solver(extendedX, y)
    return coefs[:-1], coefs[-1]


def matmul_linear_regresssion(X, y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


def batches(X, batch_size):
    for batch_start in range(0, X.shape[0], batch_size):
        yield X[batch_start: batch_start + batch_size]


def gd_linear_regresssion(X, y):
    max_running_time_seconds = 0.1
    last_time = time.time() + max_running_time_seconds
    a = np.zeros((X.shape[1],))
    eta = 0.001
    while time.time() < last_time:
        gradient = 2 * np.dot(X.T, np.dot(X, a)) - 2 * np.dot(X.T, y)
        a -= eta * gradient
    return a


def main():
    np.random.seed(123)
    n_samples = 50
    d = 10
    X = np.random.random((n_samples, d))
    a = np.random.random((d,))
    b = np.random.normal(0, 0.01, (n_samples,)) + 100
    y = np.dot(X, a) + b

    sk_a, sk_b = sk_linear_regeression(X, y)
    matmul_a, matmul_b = wrap_linear_regression_solver(X, y, matmul_linear_regresssion)
    gd_a, gd_b = wrap_linear_regression_solver(X, y, gd_linear_regresssion)
    print("sk_a", sk_a)
    print("sk_b", sk_b)
    print("matmul_a", matmul_a)
    print("matmul_b", matmul_b)
    print("gd_a", gd_a)
    print("gd_b", gd_b)


if __name__ == '__main__':
    main()


def ssto(y):
    return np.sum((y - np.average(y)) ** 2)


def sse(X, y, a, b):
    return np.sum(residuals(X, y, a, b) ** 2)


def R(X, y, a, b):
    naive_error = ssto(y)
    prediction_error = sse(X, y, a, b)
    return 1 - prediction_error / naive_error