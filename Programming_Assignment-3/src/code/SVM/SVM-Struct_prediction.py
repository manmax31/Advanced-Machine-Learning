__author__ = "Libo Yin"
__author__ = "Manab Chetia"
"""
Q 3.
This script prints out the letter and word wise accuracy based on models created by SVM_hmm
"""

import string
import numpy as np


def get_int_labels(str_labels):
    """
    This function converts a,b,c,d,...,z to 1,2,3,4,...,26
    :param  str_labels : the second column of the train or test files
    :return int_labels : integer values of the letters
    :rtype : list"""

    int_labels = []

    for str_label in str_labels:
        int_labels.append( string.ascii_lowercase.index(str_label) + 1 )

    return int_labels


def get_labels_and_wordindex(file1, file2):
    """
    This function takes in the test.txt file and outtags file and returns the correct and predicted letters
    :param file1: test.txt file
    :param file2: outtags file
    :return orig_labels: list containing the correct letters
    :return pred_labels: list containing the predicted letters
    :return wi: list of indices where a word ends i.e. the list of indices where it found "-1" which denotes end of word
    :rtype : list, list, list
    """
    data = np.loadtxt( file1, usecols=[1,2], dtype=str )

    str_labels = data[:,0]
    next_words = data[:,1]

    wi = np.insert(np.where(next_words=="-1"), 0, 0)

    orig_labels = get_int_labels( str_labels )
    pred_labels = np.loadtxt( file2 )

    return orig_labels, pred_labels, wi


def get_words(orig_labels, pred_labels, wi):
    """
    This function returns the correct words and predicted words
    :param orig_labels: list containing the correct letters
    :param pred_labels: list containing the predicted letters
    :param wi: list of indices where a word ends i.e. the list of indices where it found "-1" which denotes end of word
    :return orig_words: a list containing the correct test labels
    :return pred_words: a list containing the predicted labels
    :rtype : list, list, list"""

    orig_words = []
    pred_words = []

    start_index = 0
    for i in xrange(0, len(wi) -1):
        end_index  = wi[i + 1]

        o_word = orig_labels[start_index:end_index + 1]
        p_word = pred_labels[start_index:end_index + 1]

        start_index = end_index + 1

        orig_words.append(o_word)
        pred_words.append(p_word)

    return orig_words, pred_words


def get_letter_accuracy(orig_labels, pred_labels):
    """
    This function calculates the letter wise accuracy
    :param orig_labels: list containing the correct letters
    :param pred_labels: list containing the predicted labels
    :return: percentage of correct letters predicted
    :rtype : float
    """
    true_count = 0.0
    for l1,l2 in zip(orig_labels, pred_labels.tolist()):
        if l1==l2:
            true_count += 1
    return true_count*100/len(orig_labels)


def get_word_accuracy(orig_words, pred_words):
    """
    This function calculates the word wise accuracy
    :param orig_words: list containing correct labels
    :param pred_words: list containing predicted labels
    :return percentage of correct words predicted
    :rtype : float"""

    trueCount = 0.0
    for w1,w2 in zip(orig_words, pred_words):
        if w1==w2.tolist():
            trueCount += 1

    return trueCount*100/len(orig_words)


def main():
    """ Execution begins here """
    C = [1, 10, 50, 100, 500, 1000, 5000]
    #c = 1000
    file_str    = "../../data/" + "test.txt"

    for c in C:
        file_pred   = "outtags/" + "test" + str(c) + ".outtags"

        orig_labels, pred_labels, word_index = get_labels_and_wordindex(file_str, file_pred)

        orig_words,  pred_words              = get_words(orig_labels, pred_labels, word_index)

        print("C = {} Letter wise accuracy: {} %".format(c, get_letter_accuracy(orig_labels, pred_labels)))
        print("C = {} Word wise accuracy  : {} %".format(c, get_word_accuracy(orig_words, pred_words)))
        print

if __name__ == "__main__": main()

# C = 1 Letter wise accuracy: 65.2225360715 %
# C = 1 Word wise accuracy  : 13.4050596104 %
#
# C = 10 Letter wise accuracy: 75.0438964806 %
# C = 10 Word wise accuracy  : 26.4902587962 %
#
# C = 50 Letter wise accuracy: 80.7504389648 %
# C = 50 Word wise accuracy  : 37.888921198 %
#
# C = 100 Letter wise accuracy: 82.3459806092 %
# C = 100 Word wise accuracy  : 41.4364640884 %
#
# C = 500 Letter wise accuracy: 84.1247423467 %
# C = 500 Word wise accuracy  : 46.2925268974 %
#
# C = 1000 Letter wise accuracy: 84.7545614169 %
# C = 1000 Word wise accuracy  : 47.746437918 %
#
# C = 5000 Letter wise accuracy: 85.1668066265 %
# C = 5000 Word wise accuracy  : 49.3166618203 %