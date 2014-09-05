'''
Created on Sep 2, 2014

@author: manabchetia

This script implements BELIEF PROPAGATION in a BINARY TREE
'''

import numpy as np
import time  as t

# Probabilities
p1     = np.array(   [0.95, 0.05]                 )
pn_n_1 = np.array( [ [0.95, 0.05], [0.05, 0.95] ] )


def forwardPass(n):
    '''This function does forward message passing from rootNode x1 to all other nodes
    @param n: number of Nodes
    @return : dictionary which stores forwarded message at each node   {node1: Message, ..., nodeN: Message}
    '''
    forwardMessages    = {}
    forwardMessages[1] = p1
    forwardMessages[2] = np.dot( forwardMessages[1], pn_n_1 )
    forwardMessages[3] = np.dot( forwardMessages[1], pn_n_1 )
    for i in range(4, n+1):
        forwardMessages[i] = np.dot( forwardMessages[i/2], pn_n_1 )
    return forwardMessages


def backwardPass(node):
    ''' NOT MANDATORY as it returns [ 1   1 ]
    This function does backward message passing
    @param node : node from which backward message passing starts
    @return     : dictionary which stores backwarded message at each node  
    '''
    backwardMessages         = {}
    entirePath               = getBackwardPath(node) 
    remainingPath            = entirePath[2:]     
    backwardMessages[node]   = np.array( [1.0, 1.0] )
    backwardMessages[node/2] = np.dot  ( backwardMessages[node], pn_n_1)
    for i in remainingPath: 
        if (i*2) in backwardMessages:
            backwardMessages[i] = np.dot(backwardMessages[ i*2 ],   pn_n_1)
        else:
            backwardMessages[i] = np.dot(backwardMessages[(i*2)+1], pn_n_1)
    return backwardMessages


def getMarginalProbabilities(n, L):
    '''This function calculates the marginal probability of each node by combining messages
    from both forward and backward passes
    @param  n : Chain Length
    @return   : dictionary contain marginal probability of each node {x1: p(x1), ..., xn: p(xn)}
    ''' 
    marginalProbabilities = {}
    forwardMessages       = forwardPass(n)
    backwardMessageList   = [ backwardPass(node) for node in getNodes(L) ]
    backwardMessages      = {}
    
    # Multiply all backward messages
    for node in range(1, n+1):
        belief = np.ones([1, pn_n_1.shape[0]], dtype=np.float16)#np.array( [1.0, 1.0] )
        for backwardMessage in backwardMessageList:
            if node in backwardMessage:
                belief = np.multiply(belief, backwardMessage[node])
        backwardMessages[node] = belief
    
    # Multiply forward and backward messages
    for node in range(1, n+1):
        marginalProbabilities[node] = np.multiply(forwardMessages[node], backwardMessages[node])
    
    return marginalProbabilities
        
    
def getBackwardPath(n):
    '''This function gives us all the nodes that a message has to pass through to reach the root node (1)
    @param n: index of the node
    @return : a list of all the nodes that a message has to pass to reach root node. e.g. If n=10, it will return [10,5,2,1]
    '''
    path = []
    while n >= 1:
        path.append(n)
        n /= 2
    return path

        
def getNodes(L):
    '''This function gets the Indices of all nodes at a particular layer L
    @param L: layer for which we require all the indices of the nodes
    @return : a list of indices of that layer. e.g. If L = 2, it will return [2,3]. If L = , it will return [4,5,6,7]   
    '''
    layer = {}
    index = 1
    for i in xrange(0, L):
        indices = []
        for j in xrange(2**i):
            indices.append(index)
            index += 1
        layer[i+1] = indices
    return layer[L]
        


def printProbabitliesSmall(L, n, node):
    '''This function prints the marginal probability of a node
    @param L     : number of layers
    @param n     : number of nodes
    @param node  : the node whose marginal probability is required  
    '''
    marginalProbabilities = getMarginalProbabilities(n, L) # Get Marginal Probabilities of each node in chain as {x1:[p(x1=1), p(x1=0)], ..., xn:[p(xn=1), p(xn=0)]}
    marginalProbability   = marginalProbabilities[node]
    
    print("p( x8=1 )                      : {}".format(marginalProbability[0][0]))
    print("p( x8=1 | x12=1 )              : Not Applicable")
    print("p( x8=1 | x12=1, x7=1 )        : Not Applicable")
    print("p( x8=1 | x12=1, x7=1, x15=0 ) : Not Applicable")


def printProbabitliesLarge(L, n, node):
    '''This function prints the marginal probability of a node
    @param L     : number of layers
    @param n     : number of nodes
    @param node  : the node whose marginal probability is required  
    '''
    marginalProbabilities = getMarginalProbabilities(n, L) # Get Marginal Probabilities of each node in chain as {x1:[p(x1=1), p(x1=0)], ..., xn:[p(xn=1), p(xn=0)]}
    marginalProbability   = marginalProbabilities[node]
    
    print("p( x32=1 )                       : {}".format(marginalProbability[0][0]))
    print("p( x32=1 | x45=1 )               : Not Applicable")
    print("p( x32=1 | x45=1, x31=1 )        : Not Applicable")
    print("p( x32=1 | x45=1, x31=1, x63=0 ) : Not Applicable")   
    

def main(): 
    '''Execution Begins Here'''
    
    # Small TREE
    L            = 4             # No of Layers
    n            = 2**L - 1      # No of Nodes
    marginalNode = 8             # p(x5) i.e. Node whose marginal we require
    
    startTime = t.time()
    
    print("BELIEF PROPAGATION in a Small TREE with L = {}, n = {}".format(L, n))
    printProbabitliesSmall(L, n, marginalNode)
    
    print("Time Required                  : {} secs".format(t.time() - startTime))
    print("Execution Complete!\n")
    
    # Large TREE
    L            = 6             # No of Layers
    n            = 2**L - 1      # No of Nodes
    marginalNode = 32  # p(x5) i.e. Node whose marginal we require
    
    startTime = t.time()
    
    print("BELIEF PROPAGATION in a Large TREE with L = {}, n = {}".format(L, n))
    printProbabitliesLarge(L, n, marginalNode)
    
    print("Time Required                    : {} secs".format(t.time() - startTime))
    print("Execution Complete!")
    
    
if __name__ == "__main__" : main() 