'''
Created on Aug 30, 2014

@author: manabchetia

This script implements SAMPLING in a BINARY TREE
'''

import numpy as np
import time  as t

# Probabilities
p1   = np.array(   [0.95, 0.05]                 )
pn_2 = np.array( [ [0.95, 0.05], [0.95, 0.05] ] )


def createTree(n):
    '''This function creates a dictionary where keys represent the indices and 
    values represent the value at that index i.e. 0/1
    @param n : node of nodes in tree
    @return  : {node1: 0, node2: 0, node3: 0, ..., nodeN:0}
    '''
    tree = {x+1: 0 for x in xrange(n)}
    return tree


def sampleFrom(p, sampleX = None):
    '''
    This function generates a sample based on the transition probabilities at each step
    @param p       : transition probabilities
    @param sampleX : sample generated at previous step
    @return        : a new sample using the previous sample and transition probabilities
    '''
    sampleUniform = np.random.uniform( 0.0, 1.0, size = 1 )
    if ( sampleX == None ):
        if sampleUniform <= p[0]:
            return 1
        else:
            return 0
    else:
        if sampleUniform <= p[0]:
            return sampleX
        else:
            return 1-sampleX


def genOneSample(n, tree):
    '''This function Generates a sample from the TREE
    @param  n : no of Nodes in tree. e.g.n=15
    @return x : returns a sample that is represented by the joint Distribution
    @return   : returns the configuration (or samples) of various nodes that resulted in x
    '''
    configuration = np.empty(n)
    for i in xrange(1, n+1):
        if i==1:
            x = sampleFrom( p1, None )
        else:
            x = sampleFrom( pn_2[x], tree[i/2] )
        tree[i]            = x
        configuration[i-1] = x
    return x, configuration


def genXSamples(nSamples, n, tree):
    '''This function Generates user desired Number of Samples and the configuration that generated those Samples
    @param nSamples : no of Samples to be generated
    @param n        : no of Nodes
    @return         : desired number of Samples
    @return         : configurations that  created those Samples'''
    configurations = []            # Storage for configurations
    samples = np.empty(nSamples)
    for i in xrange(nSamples):
        samples[i], configuration = genOneSample(n, tree)
        configurations.append(configuration)
    return samples, np.array(configurations)


def printProbabilitiesSmall(configurations, nSamples):
    '''This function prints out the required probabilities as asked
    @param configurations : array of all possible configurations
    @param nSamples       : number of Samples generated  
    '''
    countSamples1 = 0.0
    countSamples2 = 0.0
    countSamples3 = 0.0
    countSamples4 = 0.0
    
    countSamplesx12_1            = 0.0
    countSamplesx12_1_x7_1       = 0.0
    countSamplesx12_1_x7_1_x15_0 = 0.0
   
    for configuration in configurations:
        if (configuration[7] == 1):
            countSamples1 += 1  
        if (configuration[7] == 1 and configuration[11] == 1):
            countSamples2 += 1
        if (configuration[7] == 1 and configuration[11] == 1 and configuration[6] == 1):
            countSamples3 += 1
        if (configuration[7] == 1 and configuration[11] == 1 and configuration[6] == 1 and configuration[14] == 0):
            countSamples4 += 1
        
        if (configuration[11] == 1):
            countSamplesx12_1 += 1
        if (configuration[11] == 1 and configuration[6] == 1):
            countSamplesx12_1_x7_1 += 1
        if (configuration[11] == 1 and configuration[6] == 1 and configuration[14] == 0):
            countSamplesx12_1_x7_1_x15_0 += 1
    
    print("p( x8=1 )                      : {}".format(countSamples1 / nSamples                     ) )
    print("p( x8=1 | x12=1 )              : {}".format(countSamples2 / countSamplesx12_1            ) )
    print("p( x8=1 | x12=1, x7=1 )        : {}".format(countSamples3 / countSamplesx12_1_x7_1       ) )
    print("p( x8=1 | x12=1, x7=1, x15=0 ) : {}".format(countSamples4 / countSamplesx12_1_x7_1_x15_0 ) )   
 
 
def printProbabilitiesLarge(configurations, nSamples):
    '''This function prints out the required probabilities as asked
    @param configurations : array of all possible configurations
    @param nSamples       : number of Samples generated  
    '''
    countSamples1 = 0.0
    countSamples2 = 0.0
    countSamples3 = 0.0
    countSamples4 = 0.0
    
    countSamplesx45_1             = 0.0
    countSamplesx45_1_x31_1       = 0.0
    countSamplesx45_1_x31_1_x63_0 = 0.0
   
    for configuration in configurations:
        if (configuration[31] == 1):
            countSamples1 += 1  
        if (configuration[31] == 1 and configuration[44] == 1):
            countSamples2 += 1
        if (configuration[31] == 1 and configuration[44] == 1 and configuration[30] == 1):
            countSamples3 += 1
        if (configuration[31] == 1 and configuration[44] == 1 and configuration[30] == 1 and configuration[62] == 0):
            countSamples4 += 1
        
        if (configuration[44] == 1):
            countSamplesx45_1 += 1
        if (configuration[44] == 1 and configuration[30] == 1):
            countSamplesx45_1_x31_1 += 1
        if (configuration[44] == 1 and configuration[30] == 1 and configuration[62] == 0):
            countSamplesx45_1_x31_1_x63_0 += 1
    
    print("p( x32=1 )                       : {}".format(countSamples1 / nSamples                      ) )
    print("p( x32=1 | x45=1 )               : {}".format(countSamples2 / countSamplesx45_1             ) )
    print("p( x32=1 | x45=1, x31=1 )        : {}".format(countSamples3 / countSamplesx45_1_x31_1       ) )
    print("p( x32=1 | x45=1, x31=1, x63=0 ) : {}".format(countSamples4 / countSamplesx45_1_x31_1_x63_0 ) )
       

def main():
    '''Execution Begins Here'''
    
    # SMALL TREE
    L = 4                  # No of Layers
    n = 2**L - 1           # No of Nodes
    nSamples = 100000      # Number of Samples required
    tree =  createTree(n)  # Creating a tree 
    
    print("Generating {} samples from a Small TREE with L = {} & n = {} ...".format(nSamples, L, n))
    
    startTime = t.time()
    
    samples, configurations = genXSamples(nSamples, n, tree)
    printProbabilitiesSmall(configurations, nSamples)
    
    print("Time Required                  : {} secs".format(t.time() - startTime))
    print("Execution Complete!\n")
    
    
    # LARGE TREE
    L = 6                  # No of Layers
    n = 2**L - 1           # No of Nodes
    nSamples = 100000      # Number of Samples required
    tree =  createTree(n)  # Creating a tree    
    
    print("Generating {} samples from a Large TREE with L = {} & n = {} ...".format(nSamples, L, n))
    
    startTime = t.time()
    
    samples, configurations = genXSamples(nSamples, n, tree)
    printProbabilitiesLarge(configurations, nSamples)
    
    print("Time Required                    : {} secs".format(t.time() - startTime))
    print("Execution Complete!")
    
    
if __name__ == '__main__':main()