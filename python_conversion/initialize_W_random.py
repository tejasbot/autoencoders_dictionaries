import numpy
from matlab_functions import *

def initialize_W_random(A):
    dW = numpy.random.randn(A.shape[0], A.shape[1])
    h = A.shape[1]
    dW = numpy.dot(normc(dW), numpy.diag(numpy.random.randint(1, 25, (h, 1))+1))
    W_T = A + dW
    W = numpy.transpose(W_T)

    return W, W_T
