import numpy
from matlab_functions

def initialize_W(A, delta):
    dW = numpy.random.randn(A.shape[0], A.shape[1])
    dW = numpy.dot(delta, normc(dW))
    W_T = A + dW
    W= numpy.transpose(W_T)
    return W, W_T
