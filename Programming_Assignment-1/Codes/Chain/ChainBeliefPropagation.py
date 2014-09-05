'''
Created on Sep 1, 2014

@author: manabchetia

This script implements BELIEF PROPAGATION in a CHAIN
'''

import numpy as np
import time  as t 

# Probabilities
p1     = np.array(   [0.95, 0.05]                 )
pn_n_1 = np.array( [ [0.95, 0.05], [0.05, 0.95] ] )


def forwardPass(n):
    '''This function does forward message passing
    @param n: number of nodes
    @return : dictionary which stores forwarded message at each node  
    '''
    forwardMessages    = {}
    forwardMessages[1] = p1
    forwardMessages[2] = np.dot(forwardMessages[1], pn_n_1)
    for i in range(3, n+1):
        forwardMessages[i] = np.dot(forwardMessages[i-1], pn_n_1)
    return forwardMessages
    

def backwardPass(n):
    '''This function does backward message passing
    @param n: number of nodes
    @return : dictionary which stores backwarded message at each node  
    '''
    backwardMessages      = {}
    backwardMessages[n]   = np.array( [1.0, 1.0] )
    backwardMessages[n-1] = np.dot  ( backwardMessages[n],   pn_n_1 )
    for i in range(n-2, 0, -1):
        backwardMessages[i] = np.dot( backwardMessages[i+1], pn_n_1)
    return backwardMessages
    
    
def getMarginalProbabitlies(n):
    '''This function calculates the marginal probability of each node by combining messages
    from both forward and backward passes
    @param  n : Chain Length
    @return   : dictionary contain marginal probability of each node {x1: p(x1), ..., xn: p(xn)}
    ''' 
    forwardMessages  = forwardPass(n)
    backwardMessages = backwardPass(n)

    marginalProbabilities = {}
    for i in range(1, n+1):
        marginalProbabilities[i] = np.multiply(forwardMessages[i], backwardMessages[i])
    return marginalProbabilities


def main():
    '''Execution Begins Here'''
    n            = 15 # Chain Length
    marginalNode = 5  # p(x5) i.e. Node whose marginal we require
    
    startTime    = t.time()
    marginalProbabilities = getMarginalProbabitlies(n) # Get Marginal Probabilities of each node in chain as {x1:[p(x1=1), p(x1=0)], ..., xn:[p(xn=1), p(xn=0)]}
    marginalProbability   = marginalProbabilities[marginalNode]
    timeRequired = t.time() - startTime
    
    print("BELIEF PROPAGATION in Chain with n = {}".format(n))
    print("p( x5=1 )                      : {}".format(marginalProbability[0]))
    print("p( x5=1 | x1=1 )               : Not Applicable")
    print("p( x5=1 | x1=1, x10=1 )        : Not Applicable")
    print("p( x5=1 | x1=1, x10=1, x15=0 ) : Not Applicable")
    print("Time Required                  : {} seconds".format(timeRequired))
    print("Execution Complete!")


if __name__ == "__main__" : main() 