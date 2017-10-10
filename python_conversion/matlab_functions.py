import numpy

def normc(A):
    A_normalizers_mtx = numpy.transpose(numpy.kron(numpy.linalg.norm(A, axis = 1), numpy.ones((A.shape[1],1))))
    A = numpy.divide(A, A_normalizers_mtx)
    return A
