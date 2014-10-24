__author__ = 'manabchetia'

import string
import numpy as np


def get_int_labels(str_labels):
    """
    This function converts a,b,c,d,... to 1,2,3,4,...
    :rtype : list"""
    int_labels = []
    for str_label in str_labels:
        int_labels.append( string.ascii_lowercase.index(str_label) + 1 )
    return int_labels


def get_words(file1, file2):
    """
    This function takes in 2 files and returns the labels, file1: original file, file2: predicted file
    :rtype : list, list"""

    data = np.loadtxt( file1, usecols=[1,2], dtype=str )

    str_labels = data[:,0]
    next_words = data[:,1]

    wi = np.insert(np.where(next_words=="-1"), 0, 0)

    orig_labels = get_int_labels( str_labels )
    pred_labels = np.loadtxt( file2 )

    orig_words = []
    pred_words = []

    sI = 0
    for i in xrange(0, len(wi) -1):
        eWi  = wi[i + 1]

        o_word = orig_labels[sI:eWi + 1]
        p_word = pred_labels[sI:eWi + 1]

        sI = eWi + 1

        orig_words.append(o_word)
        pred_words.append(p_word)

    return orig_words, pred_words



def get_accuracy(orig_words, pred_words):
    """
    This function calculates the word wise accuracy
    :rtype : accuracy %"""

    trueCount = 0
    for w1,w2 in zip(orig_words, pred_words):
        if w1==w2.tolist():
            trueCount += 1

    return trueCount*100/len(orig_words)


def main():
    c = 1000
    file_str    = "../../data/" + "test.txt"
    file_pred   = "outtags/" + "test" + str(c) + ".outtags"

    orig_words, pred_words = get_words(file_str, file_pred)

    print "C = {} Word wise accuracy: {}%".format(c, get_accuracy(orig_words, pred_words) )



if __name__ == "__main__": main()


