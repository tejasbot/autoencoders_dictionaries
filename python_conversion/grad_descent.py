import numpy
from matlab_functions import *

def grad_descent(W_init, X, Y, eta, delta, epsilon_i, threshold, max_iter):
    grad_norm = numpy.zeros((max_iter, 1))
    W = W_init
    _iter = 0

    while _iter < max_iter:
        grad_mat = grad(W,X,Y,k, delta, epsilon_i)
        grad_norm[_iter] = numpy.linalg.norm(grad_mat, 'fro', None)
        W = W - numpy.dot(eta, grad_mat)

        if _iter == 1:
            if grad_norm[_iter] > grad_norm[1]:
                print "Reducing learning rate and restarting "
                eta = eta/3
                W = W_init
                _iter = -1
                grad_norm = numpy.zeros((max_iter, 1))

        if _iter > 1:
            if grad_norm(_iter) > grad_norm(_iter -1):
                print "changing learning rate"
                eta = eta/3

        if _iter > 0:
            if grad_norm[_iter] <= grad_norm[0] * threshold:
                break 

        _iter = _iter + 1

    final_norm = grad_norm(numpy.argwhere(grad_norm > 0))
    final_norm = final_norm[-1]
    W_final = W
