import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *

if __name__ =="__main__":
    n = 100
    num_datapoints = 1000
    H = [256]
    P = [0.02]
    
#    Y_diff_init_norm = numpy.zeros((len(H), len(P)))
#    Y_diff_final_norm = numpy.zeros((len(H), len(P)))
    
    W_reps = 1
    A_reps = 1
    theta = 0.5
    
    for (i,_h) in enumerate(H):
        for (j,p) in enumerate(P):
            k = int(numpy.ceil(_h**p))
            _high = _h ** ((1-p)/2 - theta)
            _low = _high/ (_h**(p + theta))
            delta = 1/ (_h**(2*p + theta))
            
            print "Hidden Dimension: ", _h

            for u in range(0,A_reps):
                X, Y, X_test, Y_test, A_star, coherence = data_generation(n, _h, k, num_datapoints, _low, _high)
                
                eta = 0.5
                epsilon_i = 1./3 * numpy.absolute((_high + _low)/2) *k * (delta + coherence)
                threshold = 1e-8
                max_iter = 500

                for v in range(0, W_reps):
#                    init_delta = 15.0
#
                    import ipdb
                    ipdb.set_trace()
#                    
                    W, W_T = initialize_W(A_star,5*delta)
#                    W, W_T = initialize_W_random(A_star)
                    W0 = W

#                    Y_diff_init = numpy.dot(W_T, X_test) - Y_test
#                    Y_diff_init_norm[i,j] = Y_diff_init_norm[i,j] + numpy.sum(numpy.sqrt(numpy.sum(numpy.square(Y_diff_init), axis = 0)))/Y_test.shape[1]

                    W_final, final_norm = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter, A_star)
                    print "Final Gradient Norm: ",final_norm

                    
#                    W_final_norm_T = normc(numpy.transpose(W_final))
#                    Y_diff_final = numpy.dot(W_final_norm_T, X_test) - Y_test
#                    Y_diff_final_norm[i,j] = Y_diff_final_norm[i,j] + numpy.sum(numpy.sqrt(numpy.sum(numpy.square(Y_diff_final), axis = 0)))/Y_test.shape[1]

                    diff_norm = calc_maxdiffnorm(numpy.transpose(W_final), A_star)
                    init_diff_norm = calc_maxdiffnorm(numpy.transpose(W0), A_star)
                    
#                    diff = numpy.transpose(W_final) - A_star
#                    diff_norm = numpy.zeros((A_star.shape[1], 1))
#
#                    ipdb.set_trace()
#                    for t in range(0, diff.shape[1]):
#                        diff_norm[t] = numpy.linalg.norm(diff[:,t])
#
#                    init_diff = numpy.transpose(W0) - A_star
#                    init_diff_norm = numpy.zeros((A_star.shape[1], 1))
#
#                    for t in range(0, init_diff.shape[1]):
#                        init_diff_norm[t] = numpy.linalg.norm(init_diff[:,t])


                    print max(diff_norm), max(init_diff_norm)
                    ipdb.set_trace()
#            Y_diff_init_norm[i,j] = Y_diff_init_norm[i,j] / numpy.dot(W_reps, A_reps)
#            Y_diff_final_norm[i,j] = Y_diff_final_norm[i,j] / numpy.dot(W_reps, A_reps)


#    numpy.savetxt("Y_diff_init_norm.csv", Y_diff_init_norm, delimiter = "|")
#    numpy.savetxt("Y_diff_final_norm.csv", Y_diff_final_norm, delimiter = "|")
