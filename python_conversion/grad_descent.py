import numpy
from matlab_functions import *

def grad(W, X, Y, k, epsilon_i):
    h = X.shape[0]
    n = Y.shape[1]
    N = Y.shape[2]

    grad_mat = numpy.zeros((h,n))
    W_T = numpy.transpose(W)

    for j in range(0, N):
        supp = numpy.sort(numpy.argwhere(X[:,j]!=0), axis = None)

        for i in range(0,k):
            matrix_factor = numpy.multiply(numpy.dot(numpy.transpose(W_T[:,supp[i]]), Y[:,j]) - epsilon_i, numpy.eye(n)) + numpy.dot(Y[:,j], numpy.transpose(W_T[:, supp[i]]))
            vector_factor = numpy.dot(W_T, numpy.max([0, numpy.dot(W, Y[:,j])], axis = None) ) - Y[:,j]
            squared_loss_grad = numpy.dot(matrix_factor, vector_factor)
            grad_mat[supp[i],:] = grad_mat[supp[i], :] + numpy.transpose(squared_loss_grad)

    grad_mat = 1/N * grad_mat
    return grad_mat


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
