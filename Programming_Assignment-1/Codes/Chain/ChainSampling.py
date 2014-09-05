'''
Created on Aug 30, 2014

@author: manabchetia

This script implements SAMPLING in a CHAIN
'''

import numpy as np
import time  as t


# Probabilities
p1     = np.array(   [0.95, 0.05]                 )
pn_n_1 = np.array( [ [0.95, 0.05], [0.95, 0.05] ] )


def sampleFrom(p, sampleX = None):
    '''
    This function generates a sample based on the transition probabilities at each of the chain
    @param p       : transition probabilities
    @param sampleX : sample generated at previous step of a chain
    @return        : a new sample using the previous sample and transition probabilities
    '''
    sampleUniform = np.random.uniform( 0.0, 1.0, size = 1 )
    if sampleX == None:
        if sampleUniform <= p[0]:
            return 1
        else:
            return 0
    else:   
        if sampleUniform <= p[0]:
            return sampleX
        else:
            return 1-sampleX


def genOneSample(n):
    '''This functions Generates ONE sample from the chain
    @param  n : length of chain
    @return x : returns a sample which is coming out of the chain
    @return   : returns the configuration or samples at various nodes that resulted in x
    '''
    configuration = np.empty(n)
    for i in xrange(n):
        if i==0:
            x = sampleFrom(p1, None)
        else:
            x = sampleFrom(pn_n_1[x], x) 
        configuration[i] = x
    return x, configuration


def genXSamples(nSamples, n):
    '''This function Generates user desired Number of Samples and the configuration that generated those Samples
    @param nSamples : no of Samples to be generated
    @param n        : Chain Length
    @return         : desired number of Samples
    @return         : configurations that  created those Samples'''
    configurations = []   # Storage for configurations
    samples = np.empty(nSamples)
    for i in xrange(nSamples):
        samples[i], configuration = genOneSample(n)
        configurations.append(configuration)
    return samples, np.array(configurations)


def printProbabilities(configurations, nSamples):
    '''This function calculates the required probabilities as asked
    @param configurations : list of configurations that resulted from Sampling
    @param nSamples       :  number of Samples generated'''
    countSamples1 = 0.0
    countSamples2 = 0.0
    countSamples3 = 0.0
    countSamples4 = 0.0
    
    countSamplesx1_1             = 0.0
    countSamplesx1_1_x10_1       = 0.0
    countSamplesx1_1_x10_1_x15_0 = 0.0
    
    
    
    for configuration in configurations:
        if (configuration[4] == 1):
            countSamples1 += 1  
        if (configuration[4] == 1 and configuration[0] == 1):
            countSamples2 += 1
        if (configuration[4] == 1 and configuration[0] == 1 and configuration[9] == 1):
            countSamples3 += 1
        if (configuration[4] == 1 and configuration[0] == 1 and configuration[9] == 1 and configuration[14] == 0):
            countSamples4 += 1
        
        if (configuration[0] == 1):
            countSamplesx1_1 += 1
        if (configuration[0] == 1 and configuration[9] == 1):
            countSamplesx1_1_x10_1 += 1
        if (configuration[0] == 1 and configuration[9] == 1 and configuration[14] == 0):
            countSamplesx1_1_x10_1_x15_0 += 1
    
    print("p( x5=1 )                      : {}".format(countSamples1 / nSamples                     ) )
    print("p( x5=1 | x1=1 )               : {}".format(countSamples2 / countSamplesx1_1             ) )
    print("p( x5=1 | x1=1, x10=1 )        : {}".format(countSamples3 / countSamplesx1_1_x10_1       ) )
    print("p( x5=1 | x1=1, x10=1, x15=0 ) : {}".format(countSamples4 / countSamplesx1_1_x10_1_x15_0 ) )
    
    
    
    
def main():
    '''Execution Begins Here'''
    n        = 15        # Chain Length
    nSamples = 100000    # Number of Samples required
    print("Generating {} samples from a CHAIN with n = {} ... ".format(nSamples, n))
    
    startTime = t.time()
    
    samples, configurations = genXSamples(nSamples, n)
    printProbabilities(configurations, nSamples)
    
    timeRequired = t.time() - startTime
    
    print("Time Required                  : {} secs".format(timeRequired))
    print("Execution Complete!")
   
       
if __name__ == "__main__" : main()   
          