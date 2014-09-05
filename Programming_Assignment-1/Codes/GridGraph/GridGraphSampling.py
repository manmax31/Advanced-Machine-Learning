'''
Created on Aug 30, 2014

@author: manabchetia

This script implements SAMPLING in a GRID GRAPH
'''

import numpy as np
import time  as t

# Transition Probabilities
p1   = np.array(   [0.5,  0.5 ]                                           )
pn_2 = np.array( [ [0.95, 0.05], [0.95, 0.05]                           ] )
pn_3 = np.array( [ [0.01, 0.99], [0.5,  0.5 ], [0.5, 0.5], [0.99, 0.01] ] )

def createGrid(n):
    '''This function creates a dictionary where keys represent the positions and 
    values represent the value at that position
    @param n : node of nodes in tree
    @return  : {pos1: 0, pos2: 0, pos3: 0, ..., posn:0}
    '''
    grid = {x+1: 0 for x in xrange(n)}
    return grid 


def sampleFrom(p, sampleX = None):
    '''
    This function generates a sample based on the transition probabilities at step
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


def genOneSample(n, L, grid):
    '''This function Generates ONE sample from the GRID
    @param  n : no of Nodes
    @param  L : no of Layers
    @return x : returns a sample which is coming out of the GRID
    @return   : returns the configuration of various nodes that resulted in x
    '''
    configuration = np.empty(n)
    for i in xrange(1, n+1):
        if i==1:
            x = sampleFrom( p1, None )
        elif (i <= L):
            x = sampleFrom( pn_2[x], grid[i-1] )
        elif ( (i-1)%L == 0 ):
            x = sampleFrom( pn_2[x], grid[i-L] )
        else:
            xj = grid[i-1]
            xk = grid[i-L]
            if   ( xj == 0 ) and ( xk == 0):
                index = 0
            elif ( xj == 0 ) and ( xk == 1):
                index = 1
            elif ( xj == 1 ) and ( xk == 0):
                index = 2
            else:
                index = 3
            x = sampleFrom(pn_3[index], None)
        grid[i] = x
        configuration[i-1] = x
    return x, configuration


def genXSamples(nSamples, n, L, grid):
    '''Generates user desired Number of Samples and the configuration that generated those Samples
    @param nSamples : no of Samples to be generated
    @param n        : Chain Length
    @return         : desired number of Samples
    @return         : configurations that  created those Samples'''
    configurations = []   # Storage for configurations
    samples = np.empty(nSamples)
    for i in xrange(nSamples):
        samples[i], configuration = genOneSample(n, L, grid)
        configurations.append(configuration)
    return samples, np.array(configurations)


def printProbabilitiesSmall(configurations, nSamples):
    ''' This function prints the probabilities for the Smaller Grid
    @param configurations : the configurations (or samples) that resulted from Sampling process
    @param nSamples       : number of samples generated 
    '''
    countSamples1 = 0.0
    countSamples2 = 0.0
    countSamples3 = 0.0
    countSamples4 = 0.0
    
    countSamplesx16_0            = 0.0
    countSamplesx16_0_x1_0       = 0.0
    countSamplesx16_0_x1_0_x15_0 = 0.0
   
    for configuration in configurations:
        if (configuration[5] == 1):
            countSamples1 += 1  
        if (configuration[5] == 1 and configuration[15] == 0):
            countSamples2 += 1
        if (configuration[5] == 1 and configuration[15] == 0 and configuration[0] == 0):
            countSamples3 += 1
        if (configuration[5] == 1 and configuration[15] == 0 and configuration[0] == 0 and configuration[14] == 0):
            countSamples4 += 1
            
        if (configuration[15] == 0):
            countSamplesx16_0 += 1
        if (configuration[15] == 0 and configuration[0] == 0):
            countSamplesx16_0_x1_0 += 1
        if (configuration[15] == 0 and configuration[0] == 0 and configuration[14] == 0):
            countSamplesx16_0_x1_0_x15_0 += 1
    
    print("p( x6=1 )                      : {}".format( countSamples1 / nSamples                     )  )
    print("p( x6=1 | x16=0 )              : {}".format( countSamples2 / countSamplesx16_0            )  )
    print("p( x6=1 | x16=0, x1=0 )        : {}".format( countSamples3 / countSamplesx16_0_x1_0       )  )
    print("p( x6=1 | x16=0, x1=0, x15=0 ) : {}".format( countSamples4 / countSamplesx16_0_x1_0_x15_0 )  )


def printProbabilitiesLarge(configurations, nSamples):
    ''' This function prints the probabilities for the Larger Grid
    @param configurations : the configurations (or samples) that resulted from Sampling process
    @param nSamples       : number of samples generated 
    '''
    countSamples1 = 0.0
    countSamples2 = 0.0
    countSamples3 = 0.0
    countSamples4 = 0.0
    
    countSamplesx64_0            = 0.0
    countSamplesx64_0_x1_0       = 0.0
    countSamplesx57_x64_0 = 0.0
   
    for configuration in configurations:
        if (configuration[5] == 1):
            countSamples1 += 1  
        if (configuration[5] == 1 and configuration[63] == 0):
            countSamples2 += 1
        if (configuration[5] == 1 and configuration[63] == 0 and configuration[0] == 0):
            countSamples3 += 1
        if (configuration[5] == 1 and (configuration[56]==configuration[57]==configuration[58]==configuration[59]==configuration[60]==configuration[61]==configuration[62]==configuration[63]==0)):
            countSamples4 += 1
            
        if (configuration[63] == 0):
            countSamplesx64_0 += 1
        if (configuration[63] == 0 and configuration[0] == 0):
            countSamplesx64_0_x1_0 += 1
        if (configuration[56]==configuration[57]==configuration[58]==configuration[59]==configuration[60]==configuration[61]==configuration[62]==configuration[63]==0):
            countSamplesx57_x64_0 += 1
    
    print("p( x6=1 )                      : {}".format( countSamples1 / nSamples                     )  )
    print("p( x6=1 | x64=0 )              : {}".format( countSamples2 / countSamplesx64_0            )  )
    print("p( x6=1 | x64=0, x1=0 )        : {}".format( countSamples3 / countSamplesx64_0_x1_0       )  )
    print("p( x6=1 | x57= ... =x64=0 )    : {}".format( countSamples4 / countSamplesx57_x64_0        )  )    
        

def main():
    '''Execution Begins here'''
    # SMALL GRID
    L = 4
    n = L*L
    
    nSamples = 100000      # Number of Samples required
    grid =  createGrid(n)  # Creating a grid 
    
    startTime = t.time()
    
    print("Generating {} samples from a Small GRID with L = {} & n = {} ...".format(nSamples, L, n))
    samples, configurations = genXSamples(nSamples, n, L, grid)
    printProbabilitiesSmall(configurations, nSamples)
    
    print("Time Required                  : {} secs".format(t.time() - startTime))
    print("Execution Complete!\n")
    
    # LARGE GRID
    L = 8
    n = L*L
    
    nSamples = 100000      # Number of Samples required
    grid =  createGrid(n)  # Creating a grid 
    
    startTime = t.time()
    
    print("Generating {} samples from a LARGE GRID with L = {} & n = {} ...".format(nSamples, L, n))
    samples, configurations = genXSamples(nSamples, n, L, grid)
    printProbabilitiesLarge(configurations, nSamples)
    
    print("Time Required                  : {} secs".format(t.time() - startTime))
    print("Execution Complete!")
    
    
if __name__ == '__main__':main()