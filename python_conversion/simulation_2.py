import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *

if __name__ == "__main__":
    H = [1024, 2048, 4096, 8192, 8192*2]
    delta_multipliers = [2, 50]
    no_trials = 5
    eta = 0.1
    p = 0.2
    theta = 0.5
    epsilon_i = 1e-4
    max_iter = 500
    num_datapoints = 1000
    n = 50
    
#    import ipdb
#    ipdb.set_trace()
    
    grad_vector = numpy.zeros((len(H), len(delta_multipliers)+1))
    
    for index_h, h in enumerate(H):
        dmul = delta_multipliers + [h]

        k = int(numpy.ceil(h**p))
        _high = h ** ((1-p)/2 - theta)
        _low = _high/ (h**(p + theta))
        delta = 1/ (h**(2*p + theta))
        
        
        for index_delta, d in enumerate(dmul):
            
            
            for trial in range(0, no_trials):
                X, Y, X_test, Y_test, A_star, coherence = data_generation(n, h, k, num_datapoints, _low, _high)
                W, W_T = initialize_W(A_star,d*delta)
                gradient = grad(W, X, Y, k, delta, epsilon_i)

                grad_norm = numpy.linalg.norm(gradient, 'fro', None)

                print "Grad norm for h:{h}, delta_multiplier: {d}, trial: {trial} = {grad_norm}".format(h=str(h), d = str(d), trial=str(trial), grad_norm = grad_norm)

                grad_vector[index_h, index_delta] += grad_norm


    grad_vector = grad_vector/no_trials

                
