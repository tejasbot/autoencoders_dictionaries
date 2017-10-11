import numpy
import random
from matlab_functions import *



def generate_sparse_matrix(h, k, num_points, _low, _high):
    X = numpy.random.uniform(_low, _high, (h, num_points))
    arr = numpy.transpose(numpy.vstack((numpy.zeros((h-k,num_points)), numpy.ones((k,num_points)))))
    [random.shuffle(a) for a in arr]
    supp = numpy.transpose(arr)

    X = numpy.multiply(X, supp)
    return X

def data_generation(n, h, k, num_datapoints, _low, _high):
    A_star = numpy.random.randn(n, h)
    A_star = normc(A_star)
    
    coherence_mat = numpy.dot(numpy.transpose(A_star), A_star)
    coherence = numpy.max(numpy.absolute(coherence_mat - numpy.eye(h)), axis = None)/ numpy.sqrt(n)
    num_test = int(numpy.ceil(0.05 * num_datapoints));
    num_train = num_datapoints - num_test
    import ipdb
    ipdb.set_trace()

    X_train = generate_sparse_matrix(h, k, num_train, _low, _high)
    X_test = generate_sparse_matrix(h, k, num_test, _low, _high)
    Y_train = numpy.dot(A_star, X_train)
    Y_test = numpy.dot(A_star, X_test)


    return X_train, Y_train, X_test, Y_test, A_star, coherence
