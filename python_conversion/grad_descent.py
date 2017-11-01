import numpy
from matlab_functions import *
#
#def grad(W, X, Y, k, delta, epsilon_i):
#    import ipdb
#    ipdb.set_trace()
#    h = X.shape[0]
#    n = Y.shape[0]
#    N = Y.shape[1]
#
#    grad_mat = numpy.zeros((h,n))
#    W_T = numpy.transpose(W)
#
#    for j in range(0, N):
#        supp = numpy.sort(numpy.argwhere(X[:,j]!=0), axis = None)
#
#        for i in range(0,k):
#            matrix_factor = numpy.multiply(numpy.dot(numpy.transpose(W_T[:,supp[i]]), Y[:,j]) - epsilon_i, numpy.eye(n)) + numpy.dot(Y[:,j], numpy.transpose(W_T[:, supp[i]]))
#            
#            vector_factor = numpy.dot(W_T, numpy.max([0, numpy.dot(W, Y[:,j])], axis = None) ) - Y[:,j]
#            squared_loss_grad = numpy.dot(matrix_factor, vector_factor)
#            grad_mat[supp[i],:] = grad_mat[supp[i], :] + numpy.transpose(squared_loss_grad)
#
#    grad_mat = 1/N * grad_mat
#    return grad_mat


def grad_no_support(W, X, Y, k, delta, epsilon_i):
#    import ipdb
#    ipdb.set_trace() 
    h = W.shape[0]
    n = Y.shape[0]
    N = Y.shape[1]
    grad_mat = numpy.zeros((h, n))
    for data_index in range(0, N):
        y = numpy.matrix(Y[:, data_index]).transpose()
        x = X[:, data_index]
        support = numpy.argwhere(x != 0).ravel().tolist()
        
        #computing \Sum_{j=]}^h ReLU(W_j^T - \epsilon_j)W_j - y
        _sum = 0
        for j in support: #range(0, h):
            scalar = numpy.max([0 , numpy.dot(W[j,:], y) - epsilon_i])
            _sum += numpy.matrix(scalar*numpy.transpose(W[j,:])).transpose()
        _sum -= y

        
        for i in support: #range(0, h):
            scalar_term = numpy.array(numpy.dot(W[i,:], y) - epsilon_i).squeeze()
#            scalar_term_threshold = numpy.array(1.0* ((scalar_term)>0)).squeeze()
            square_term = scalar_term * numpy.eye(n) + numpy.dot(y, numpy.matrix(W[i,:]))
            grad_mat[i, :] += numpy.array(numpy.dot(square_term, _sum)).ravel()

    grad_mat = 1./N * grad_mat
    return grad_mat


def grad(W, X, Y, k, delta, epsilon_i, data_points = []):
    h = W.shape[0]
    n = Y.shape[0]
    N = Y.shape[1]
    grad_mat = numpy.zeros((h, n))
    if not len(data_points):
        data_points = range(0, N)
    for data_index in data_points:
        y = numpy.matrix(Y[:, data_index]).transpose()
        x = X[:, data_index]
#        support = numpy.argwhere(x != 0).ravel().tolist()
        
        #computing \Sum_{j=]}^h ReLU(W_j^T - \epsilon_j)W_j - y
        _sum = 0
        for j in range(0, h):
            scalar = numpy.max([0 , numpy.dot(W[j,:], y) - epsilon_i])
            _sum += numpy.matrix(scalar*numpy.transpose(W[j,:])).transpose()
        _sum -= y

        
        for i in range(0, h):
            scalar_term = numpy.array(numpy.dot(W[i,:], y) - epsilon_i).squeeze()
            scalar_term_threshold = numpy.array(1.0* ((scalar_term)>0)).squeeze()
            square_term = scalar_term * numpy.eye(n) + numpy.dot(y, numpy.matrix(W[i,:]))
            grad_mat[i, :] += numpy.array(numpy.dot(scalar_term_threshold * square_term, _sum)).ravel()

    grad_mat = 1./len(data_points) * grad_mat
    return grad_mat


def sgd(W, X, Y, k, delta, epsilon_i, batch_size_percentage = 0.1):
    data_count = int(min([Y.shape[1], numpy.rint(batch_size_percentage * Y.shape[1])]))
    data_points = numpy.random.choice(range(0, Y.shape[1]), data_count)
    grad_mat = grad(W, X, Y, k, delta, epsilon_i, data_points = data_points)
    return grad_mat


def adam(W, X, Y, k, delta, epsilon_i, eta, _iter, batch_size_percentage = 0.1):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.001

    if _iter == 1:
        m = 0
        v = 0
    else:
        grad_mat = sgd(W, X, Y, k, delta, epsilon_i, batch_size_percentage = batch_size_percentage)
        m = beta1 * m + (1-beta1) * grad_mat
        v = beta2 * v + (1-beta2) * numpy.multiply(grad_mat, grad_mat)
        mhat = m / (1-beta1**_iter)
        vhat = v / (1-beta2**_iter)
        W = W - alpha/(numpy.sqrt(numpy.linalg.norm(vhat.ravel())) + epsilon) * mhat

    return grad_mat, W

def calc_maxdiffnorm(A, B):
    assert A.shape == B.shape
    _diff = A - B
    diffnorm = numpy.zeros((A.shape[1], ))
    for t in range(0, A.shape[1]):
        diffnorm[t] = numpy.linalg.norm(_diff[:,t])
    return max(diffnorm)

def grad_descent(W_init, X, Y, k, eta, delta, epsilon_i, threshold, max_iter, A_star, optimization_method = 'regular', batch_size_percentage = 0.1):
    grad_norm = numpy.zeros((max_iter, 1))
    diff_norm = numpy.zeros((max_iter, 1))
    W = W_init
    _iter = 0

    while _iter < max_iter:
        if optimization_method == 'regular': 
            grad_mat = grad(W,X,Y,k, delta, epsilon_i)
            W = W - numpy.dot(eta, grad_mat)
        elif optimization_method == 'sgd' and batch_size_percentage!=0:
            grad_mat = sgd(W, X, Y, k, delta, epsilon_i, batch_size_percentage = batch_size_percentage)
            W = W - numpy.dot(eta, grad_mat)
        elif optimization_method == 'adam' and batch_size_percentage!=0:
            grad_mat, W = adam(W, X, Y, k, delta, epsilon_i, eta, _iter, batch_size_percentage = batch_size_percentage)
        else:
            return None 
        grad_norm[_iter] = numpy.linalg.norm(grad_mat, 'fro', None)
        diff_norm[_iter] = numpy.linalg.norm(numpy.transpose(W)- A_star)
        
        print "iteration: ", _iter, " norm: ", grad_norm[_iter] 
        print "diff_norm: ", diff_norm[_iter]

        if optimization_method == 'regular':
            if _iter == 1:
                if grad_norm[_iter] > grad_norm[1]:
                    print "Reducing learning rate and restarting "
                    eta = eta/3
                    W = W_init
                    _iter = -1
                    grad_norm = numpy.zeros((max_iter, 1))

            if _iter > 1:
                if grad_norm[_iter] > grad_norm[_iter -1]:
                    print "changing learning rate"
                    eta = eta/3

        if _iter > 0:
            if grad_norm[_iter] <= grad_norm[0] * threshold:
                break 

        _iter = _iter + 1

#    import ipdb
#    ipdb.set_trace() 
    final_norm = grad_norm[-1]
    W_final = W
#    ipdb.set_trace() 
    return W_final, final_norm, diff_norm
