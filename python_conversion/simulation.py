import numpy
from data_generation import data_generation


if __name__ =="__main__":
    import ipdb
    ipdb.set_trace()
    n = 50
    num_datapoints = 500
    H = [8]
    P = [0.2]
    
    Y_diff_init_norm = numpy.zeros((len(H), len(P)))
    Y_diff_final_norm = numpy.zeros((len(H), len(P)))
    
    W_reps = 3
    A_reps = 3

    for _h in H:
        for p in P:
            k = int(numpy.ceil(_h**p))
            m1 = -1 * _h ** (-1.5)
            print "Hidden Dimension: ", _h

            for u in range(0,A_reps):
                var = data_generation(n, _h, k, num_datapoints, m1)
            
