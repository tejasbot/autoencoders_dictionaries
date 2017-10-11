import numpy

def normc(A):
    A_normalizers_mtx = numpy.kron(numpy.linalg.norm(A, axis = 0), numpy.ones((A.shape[0],1 )))
    A = numpy.divide(A, A_normalizers_mtx)
    return A
