import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *

if __name__ =="__main__":
#    import ipdb
#    ipdb.set_trace()
    n = 50
    num_datapoints = 10
    H = [256]
    P = [0.2]
    
    Y_diff_init_norm = numpy.zeros((len(H), len(P)))
    Y_diff_final_norm = numpy.zeros((len(H), len(P)))
    
    W_reps = 3
    A_reps = 3
    theta = 0.0001
    for (i,_h) in enumerate(H):
        for (j,p) in enumerate(P):
            k = int(numpy.ceil(_h**p))
            _high = _h ** ((1-p)/2 - theta)
            _low = _high/ (_h**(p + theta))
            delta = 1/ (_h**(2*p + theta))
            
            print "Hidden Dimension: ", _h

            for u in range(0,A_reps):
                X, Y, X_test, Y_test, A_star, coherence = data_generation(n, _h, k, num_datapoints, _low, _high)
                
                eta = 0.9
                import ipdb
                ipdb.set_trace() 
                epsilon_i = 1./2 * numpy.absolute((_high + _low)/2) *k * (delta + coherence)
                threshold = 1e-8
                max_iter = 100

                for v in range(0, W_reps):
                    init_delta = 15.0
#                    W, W_T = initialize_W(A_star, init_delta)
                    W, W_T = initialize_W_random(A_star)
                    W0 = W

                    Y_diff_init = numpy.dot(W_T, X_test) - Y_test
                    Y_diff_init_norm[i,j] = Y_diff_init_norm[i,j] + numpy.sum(numpy.sqrt(numpy.sum(numpy.square(Y_diff_init), axis = 0)))/Y_test.shape[1]

#                    import ipdb
#                    ipdb.set_trace() 
                    W_final, final_norm = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter)
                    print "Final Gradient Norm: ",final_norm

                    
                    W_final_norm_T = normc(numpy.transpose(W_final))
                    Y_diff_final = numpy.dot(W_final_norm_T, X_test) - Y_test
                    Y_diff_final_norm[i,j] = Y_diff_final_norm[i,j] + numpy.sum(numpy.sqrt(numpy.sum(numpy.square(Y_diff_final), axis = 0)))/Y_test.shape[1]
                    
                    diff = numpy.transpose(W_final) - A_star
                    diff_norm = numpy.zeros((A_star.shape[1], 1))

                    for t in range(0, diff.shape[1]):
                        diff_norm[t] = numpy.linalg.norm(diff[:,t], axis = 1)

                    init_diff = numpy.transpose(W0) - A_star
                    init_diff_norm = numpy.zeros((A_star.shape[1], 1))

                    for t in range(0, init_diff.shape[1]):
                        init_diff_norm[t] = numpy.linalg.norm(init_diff[:,t], axis = 1)


            Y_diff_init_norm[i,j] = Y_diff_init_norm[i,j] / numpy.dot(W_reps, A_reps)
            Y_diff_final_norm[i,j] = Y_diff_final_norm[i,j] / numpy.dot(W_reps, A_reps)


    numpy.savetxt("Y_diff_init_norm.csv", Y_diff_init_norm, delimiter = "|")
    numpy.savetxt("Y_diff_final_norm.csv", Y_diff_final_norm, delimiter = "|")
