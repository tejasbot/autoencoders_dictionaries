import numpy

def data_generation(n, h, k, num_datapoints, m1):
    A_star = numpy.randn((n, h))
    A_star = numpy.linalg.norm(A_star, axis = 1)
    coherence_mat = numpy.transpose(A_star) * A_star
    coherence = max(max(numpy.abs(coherence_mat - numpy.eye(h))))/ numpy.sqrt(n)
    num_test = numpy.ceil(0.05 * num_datapoints);
    num_train = num_datapoints - num_test

    X = numpy.zeros((h, num_train))
    X_test = numpy.zeros((h, num_test))

    var_x_star = 1/256

    X_k = numpy.normrnd(m1, var_x_star, (k, num_datapoints))
    supp = numpy.sort(numpy.random.randint(0, high=h-1, size=k))

    # fill in X_train

    # fill in X_test

    Y = A_star * X
    Y_test = A_star * X_test

    m2 = var_x_star + m1^2

    return (X, Y, X_test, Y_test, A_star, coherence, m2)
