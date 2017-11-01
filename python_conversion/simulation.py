import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *
import json
import sys
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=100, type = int)
    parser.add_argument('--num_datapoints', default = 5000, type = int)
    parser.add_argument('--h', default = 256, type = int)
    parser.add_argument('--p', default = 0.1, type = float)
    parser.add_argument('--optimization_method', default = 'sgd')
    parser.add_argument('--d', default=50, type = int)
    parser.add_argument('--batch_size_percentage', default = 0.05, type = float)
    args = parser.parse_args()

    n = args.n
    num_datapoints = args.num_datapoints
    _h = args.h
    p = args.p
    optimization_method = args.optimization_method
    initd = args.d
    batch_size_percentage = args.batch_size_percentage
    
#    n = 100
#    num_datapoints = 1000
#    H = [1024]
#    P = [0.1]
#    optimization_method='sgd'
    
#    Y_diff_init_norm = numpy.zeros((len(H), len(P)))
#    Y_diff_final_norm = numpy.zeros((len(H), len(P)))
    
    W_reps = 1
    A_reps = 1
    theta = 0.5
    _high = 10
    _low = 1
    
#    for (i,_h) in enumerate(H):
#        for (j,p) in enumerate(P):
    k = int(numpy.ceil(_h**p))
    _high = _h ** ((1-p)/2 - theta)
    _low = _high/ (_h**(p + theta))
    X, Y, X_test, Y_test, A_star, coherence = data_generation(n, _h, k, num_datapoints, _low, _high)
    delta = 1/ (_h**(2*p + theta))
    eta = 0.5
    epsilon_i = 1./3 * numpy.absolute((_high + _low)/2) *k * (delta + coherence)
    threshold = 1e-3
    max_iter = 2500
    
    print "Hidden Dimension: ", _h

    for u in range(0,A_reps):
         
        W, W_T = initialize_W(A_star,initd*delta)
        W0 = W

        W_final, final_norm, diff_norm = grad_descent(W, X, Y, k, eta, delta, epsilon_i, threshold, max_iter, A_star, optimization_method='sgd', batch_size_percentage = batch_size_percentage)
        print "Final Gradient Norm: ",final_norm
        

        final_diff_norm = calc_maxdiffnorm(numpy.transpose(W_final), A_star)
        init_diff_norm = calc_maxdiffnorm(numpy.transpose(W0), A_star)

        result = {'h': _h, 'p': p, 'diff_norm': diff_norm.tolist(), 'init_diff_norm': init_diff_norm, 'final_diff_norm': final_diff_norm}

        fname = 'h-{h}_p-{p}_u-{u}_initdelta-{initd}_opt-{opt}.json'.format(h=_h, p=p, u=u, initd = initd, opt = optimization_method)     
        print fname
        with open(fname, 'w') as outf: 
            json.dump(result, outf)
        
        print final_diff_norm, init_diff_norm


if __name__ =="__main__":
    sys.exit(main())

#    numpy.savetxt("Y_diff_init_norm.csv", Y_diff_init_norm, delimiter = "|")
#    numpy.savetxt("Y_diff_final_norm.csv", Y_diff_final_norm, delimiter = "|")
