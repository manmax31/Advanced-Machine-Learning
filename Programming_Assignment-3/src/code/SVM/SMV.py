__author__ = 'manabchetia'
#from sklearn.svm.liblinear import fit
import numpy as np
import string


def get_int_labels(str_labels):
    int_labels = []
    for str_label in str_labels:
        int_labels.append( string.ascii_lowercase.index(str_label) + 1 )
    return int_labels


def get_labels(file1, file2):
    str_labels  = np.loadtxt( file1, usecols=[1], dtype=str )
    orig_labels = get_int_labels(str_labels)
    pred_labels = np.loadtxt( file2 )
    return orig_labels, pred_labels


def get_accuracy(orig_labels, pred_labels):
    true_count = len( np.where( (orig_labels==pred_labels) == True )[0] )
    return true_count*100/len(orig_labels)


def main():
    c = 500
    file_str    = "../../data/"    + "test.txt"
    file_pred   = "../../outtags/" + "test" + str(c) + ".outtags"

    orig_labels, pred_labels = get_labels(file_str, file_pred)


    print "C = {} Letter wise accuracy: {}%".format(c, get_accuracy(orig_labels, pred_labels) )


if __name__ == "__main__": main()


