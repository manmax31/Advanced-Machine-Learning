'''
Created on Aug 26, 2014

@author: manabchetia

This script implements BRUTE FORCE in a CHAIN
'''
import itertools as it
import numpy     as np
import time      as t


def getTransitionProb(i, j):
    '''This function calculates the transition probabilities
    @param   i: node i
    @param   j: node j
    @return   :  transition probability'''
    if (i==j):
        return 0.95
    else:
        return 0.05


def getJointProbablity(configuration):
    '''This function calculates the joint probability p(x1,...,xn) for a particular configuration
    @param : configuration e.g.[010] or [101]
    @return: joint probability of the configuration e.g.p([010]) or p([101])   
    '''
    x1 = configuration[0]
    if(x1) == 0:
        p_x1 = 0.05
    else:
        p_x1 = 0.95
    pii = 1.0
    n   = configuration.size
    for j in xrange(1, n):
        i   = configuration[j]
        i_1 = configuration[j-1]
        pii *= getTransitionProb(i, i_1)
    prob = p_x1 * pii
    return prob


def getTotalProbability(configurations):
    '''NOT REQUIRED
    This function calculates the total Probability as a check that probabilities sum up to 1
    @param: configurations = [[00]
                               [01]
                               [10]
                               [11]]
    @return: Total Probability
    '''
    totalProbability = 0.0
    for configuration in configurations:
        jointProbability  = getJointProbablity(configuration)
        totalProbability += jointProbability
        print(configuration, "  ", jointProbability)
    return totalProbability


def printProbabilities(configurations):
    '''This function prints out the required probabilities as asked
    @param configurations: array of all possible configurations
    '''
    prob1 = 0.0
    prob2 = 0.0
    prob3 = 0.0
    prob4 = 0.0
    
    probx1_1             = 0.0
    probx1_1_x10_1       = 0.0
    probx1_1_x10_1_x15_0 = 0.0
    
    
    for configuration in configurations:
        if (configuration[4] == 1):
            prob1 += getJointProbablity(configuration)
        if (configuration[4] == 1 and configuration[0] == 1):
            prob2 += getJointProbablity(configuration)
        if (configuration[4] == 1 and configuration[0] == 1 and configuration[9] == 1):
            prob3 += getJointProbablity(configuration)
        if (configuration[4] == 1 and configuration[0] == 1 and configuration[9] == 1 and configuration[14] == 0):
            prob4 += getJointProbablity(configuration)
            
        if (configuration[0] == 1):
            probx1_1 += getJointProbablity(configuration)
        if (configuration[0] == 1 and configuration[9] == 1):
            probx1_1_x10_1 += getJointProbablity(configuration)
        if (configuration[0] == 1 and configuration[9] == 1 and configuration[14] == 0):
            probx1_1_x10_1_x15_0 += getJointProbablity(configuration)
            
        
    
    print("p( x5=1 )                      : {}".format(prob1                        ) )
    print("p( x5=1 | x1=1 )               : {}".format(prob2 / probx1_1             ) )
    print("p( x5=1 | x1=1, x10=1 )        : {}".format(prob3 / probx1_1_x10_1       ) )
    print("p( x5=1 | x1=1, x10=1, x15=0 ) : {}".format(prob4 / probx1_1_x10_1_x15_0 ) )


def main():
    '''Execution Begins Here'''
    n = 15                   # Chain Length
    print("BRUTE FORCE Chain n = {}".format(n))
    startTime = t.time()
    configurations = np.asarray(list(it.product([0,1], repeat=n)))
    printProbabilities(configurations)
    print("Time Required                  : {} secs".format( t.time() - startTime ) )
    print("Execution Complete!")
   
    
if __name__ == "__main__" : main()  