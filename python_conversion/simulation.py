import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *
import json

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
    _high = 10
    _low = 1
    
    for (i,_h) in enumerate(H):
        for (j,p) in enumerate(P):
            k = int(numpy.ceil(_h**p))
#            _high = _h ** ((1-p)/2 - theta)
#            _low = _high/ (_h**(p + theta))
            X, Y, X_test, Y_test, A_star, coherence = data_generation(n, _h, k, num_datapoints, _low, _high)
            delta = 1/ (_h**(2*p + theta))
            eta = 0.5
            epsilon_i = 1./3 * numpy.absolute((_high + _low)/2) *k * (delta + coherence)
            threshold = 1e-3
            max_iter = 300
            
            print "Hidden Dimension: ", _h

            for u in range(0,A_reps):
                 
                W, W_T = initialize_W(A_star,5*delta)
                W0 = W

                W_final, final_norm, diff_norm = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter, A_star)
                print "Final Gradient Norm: ",final_norm
                

                final_diff_norm = calc_maxdiffnorm(numpy.transpose(W_final), A_star)
                init_diff_norm = calc_maxdiffnorm(numpy.transpose(W0), A_star)

                result = {'h': _h, 'p': p, 'diff_norm': diff_norm.tolist(), 'init_diff_norm': init_diff_norm, 'final_diff_norm': final_diff_norm}

                fname = 'h-' + str(_h) + '_p-' + str(p) + '_u-' + str(u)+'.json'
                print fname
                with open(fname, 'w') as outf: 
                    json.dump(result, outf)
                
                print final_diff_norm, init_diff_norm


#    numpy.savetxt("Y_diff_init_norm.csv", Y_diff_init_norm, delimiter = "|")
#    numpy.savetxt("Y_diff_final_norm.csv", Y_diff_final_norm, delimiter = "|")
