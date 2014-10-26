__author__ = 'manabchetia'


from scipy.misc import imrotate
import numpy as np
import string

def rotation(X, alpha):
    Y = imrotate(X, alpha)
    len_x1, len_x2 = X.shape
    len_y1, len_y2 = Y.shape

    from_x = np.floor((len_y1 + 1 - len_x1)/2)
    from_y = np.floor((len_y2 + 1 - len_x2)/2)

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


def get_int_labels(str_labels):
    """
    This function converts a,b,c,d,... to 1,2,3,4,...
    :rtype : list"""
    int_labels = []
    for str_label in str_labels:
        int_labels.append( string.ascii_lowercase.index(str_label) + 1 )
    return int_labels


def get_words_values(file_name):
    """
    This function takes in 2 files and returns the labels, file1: original file, file2: predicted file
    :rtype : list, list"""
    labels__next_words = np.loadtxt( file_name, usecols=[1,2], dtype=str)
    values_all         = np.loadtxt( file_name, usecols=range(5,128+5))

    str_labels = labels__next_words[:,0]
    next_words = labels__next_words[:,1]

    wi = np.insert(np.where(next_words=="-1"), 0, 0)

    int_labels = get_int_labels( str_labels )

    words        = []
    words_values = []

    sI = 0
    for i in xrange(0, len(wi) -1):
        eI  = wi[i + 1]

        word   = int_labels[sI:eI + 1]
        values = values_all[sI:eI + 1]

        sI = eI + 1

        words.append(word)
        words_values.append(values)

    return words, words_values


def main():
    train_file   = "../../data/" + "train.txt"
    test_file    = "../../data/" + "test.txt"
    command_file = "../../data/" + "transform.txt"

    commands = open(command_file).readlines()

    train_words, train_words_values = get_words_values(train_file)

    for command in commands:
        comm_arr    = command.strip().split(" ")
        word_index  = comm_arr[1] - 1
        word_values =
        if comm_arr[0] == 'r':







if __name__ == "__main__": main()




