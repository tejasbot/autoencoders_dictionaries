import numpy
from data_generation import data_generation
from matlab_functions import *
from initialize import *
from grad_descent import *

if __name__ == "__main__":
    H = [1024 2048 4096 8192 8192*2]
    init_Delta = [2 50]
    no_trials = 5
    eta = 0.1
    p = 0.2
    theta = 0.5
    epsilon_i = 1e-4
    max_iter = 100

    grad_vector = numpy.zeros((len(H), len(init_Delta)+1))
    
    for index_h, h in enumerate(H):
        init_delta = init_Delta + [h]

        k = int(numpy.ceil(h**p))
        _high = h ** ((1-p)/2 - theta)
        _low = _high/ (h**(p + theta))

        
        
        for index_delta, delta in enumerate(init_delta):
            
            
            for trial in no_trials:
                X, Y, X_test, Y_test, A_star, coherence = data_generation(n, h, k, num_datapoints, _low, _high)
                W, W_T = initialize_W(A_star,h*delta)
                grad = grad_no_support(W, X, Y, k, delta, epsilon_i)

                grad_norm = numpy.linalg.norm(grad, 'fro', None)

                print "Grad norm for h:{h}, delta: {delta}, trial: {trial} = {grad_norm}".format(h=str(h), delta = str(delta), trial=str(trial))

                grad_norm[index_h, index_delta] += grad_norm


    grad_norm = grad_norm/no_trials

                
