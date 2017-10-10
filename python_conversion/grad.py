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
            matrix_factor = numpy.multiply(numpy.dot(numpy.transpose(W_T[:,supp[i]]), Y[;,j]) - epsilon_i, numpy.eye(n)) + numpy.dot(Y[:,j], numpy.transpose(W_T[:, supp[i]]))
            vector_factor = numpy.dot(W_T, numpy.max([0, numpy.dot(W, Y[:,j])], axis = None) ) - Y[:,j]
            squared_loss_grad = numpy.dot(matrix_factor, vector_factor)
            grad_mat[supp[i],:] = grad_mat[supp[i], :] + numpy.transpose(squared_loss_grad)

    grad_mat = 1/N * grad_mat
    return grad_mat
