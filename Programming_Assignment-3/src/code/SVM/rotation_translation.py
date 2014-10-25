__author__ = 'manabchetia'


from scipy.misc import imrotate
import numpy as np

def rotation(X, alpha):
    Y = imrotate(X, alpha)
    lenx1, lenx2 = X.shape
    leny1, leny2 = Y.shape

    fromx = np.floor((leny1 + 1 - lenx1)/2)
    fromy = np.floor((leny2 + 1 - lenx2)/2)

    idx = np.where(Y==0)
    Y[idx] = X[idx]

    return Y

def translation(X, offset):
    Y = X
    lenx, leny = X.shape
    ox = offset[0]
    oy = offset[1]

    # # General case where ox and oy can be negative
    Y[ max(1,1+ox):min(lenx, lenx+ox), max(1,1+oy):min(leny, leny+oy) ] = X[max(1,1-ox):min(lenx, lenx-ox), max(1,1-oy):min(leny, leny-oy)]

    # # Special case where ox and oy are both positive (used in this project)
    # Y[1+ox:lenx, 1+oy:leny] = X[1:lenx-ox, 1:leny-oy]
    return Y

def get_X_Y(file):
    """
    This function extract the Y, X from the datasets
    :rtype : list, list
    """
    str_labels = np.loadtxt( file, usecols=[1], dtype=str)
    X          = np.loadtxt( file, usecols=range(5, 128+5)).tolist()
    Y          = get_int_labels( str_labels )
    return X, Y

def main():

if __name__ == "__main__": main()




