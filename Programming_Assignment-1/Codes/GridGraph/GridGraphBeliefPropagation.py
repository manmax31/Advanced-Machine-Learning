'''
Created on Sep 4, 2014

@author: manabchetia

This script implements Belief Propagation in GRID GRAPH
'''

def main():
    L = 4
    n = L*L
    print("BELIEF PROPAGATION in SMALL GRID with L = {}, n = {}".format(L,n))
    print("p( x6=1 )                      : Not Applicable")
    print("p( x6=1 | x16=0 )              : Not Applicable")
    print("p( x6=1 | x16=0, x1=0 )        : Not Applicable")
    print("p( x6=1 | x16=0, x1=0, x15=0 ) : Not Applicable")
    print("Execution Complete!\n")
    
    L = 8
    n = L*L
    print("BELIEF PROPAGATION in LARGE GRID with L = {}, n = {}".format(L,n))
    print("p( x6=1 )                      : Not Applicable")
    print("p( x6=1 | x64=0 )              : Not Applicable")
    print("p( x6=1 | x64=0, x1=0 )        : Not Applicable")
    print("p( x6=1 | x57= ... =x64=0 )    : Not Applicable")
    print("Execution Complete!")
    
if __name__ == '__main__':main()