import sobol
import random
import numpy as np

def neuron_genome_samp(n,solution,width):
    '''generate quasi-random set of n points in 12D par space (no time constants) centered on the solution in question extended 
    out by width in all directions'''
    genelen = len(solution)
    CTRNNsize = int(np.sqrt(1+genelen)-1)
    dim = genelen-CTRNNsize
    samp = sobol.sample(dimension=dim, n_points=n, skip=random.randint(0,5000)) #set between 0 and 1 in all dimensions
    samp = (samp - .5) * 2 #center around 0 and scale back up to [-1,1]
    samp = samp * width
    samp = samp + solution[:-CTRNNsize]
    sample = np.zeros((n,genelen))
    sample[:,dim:] = solution[dim:]
    sample[:,:dim] = samp
    return sample