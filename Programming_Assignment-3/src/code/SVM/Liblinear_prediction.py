__author__ = "Libo Yin"
__author__ = "Manab Chetia"
"""
Q 3.
This script implements SVM MC, creates a model from the training data and finally prints out the accuracy on the test data set
"""

import liblinearutil as lu
import numpy as np
import string


def get_int_labels(str_labels):
    """
    This function converts a,b,c,d,...,z to 1,2,3,4,...,26
    :param  str_labels : the second column of the train or test files
    :return int_labels : integer values of the letters
    :rtype : list
    """
    int_labels = []
    for str_label in str_labels:
        int_labels.append( string.ascii_lowercase.index(str_label) + 1 )
    return int_labels


def get_X_Y_wi(file):
    """
    This function extract the Y, X from the datasets
    :param file : training or test file
    :return X   : features
    :return Y   : labels
    :rtype : list, list
    """
    data       = np.loadtxt( file, usecols=[1,2], dtype=str)
    str_labels = data[:,0]
    next_words = data[:,1]

    wi = np.insert(np.where(next_words=="-1"), 0, 0)

    X          = np.loadtxt( file, usecols=range(5, 128+5)).tolist()
    Y          = get_int_labels( str_labels )
    return X, Y, wi


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


def train(c, Y_train, X_train):
    """
    This function takes in the training labels and features and creates a model and saves that model
    :param C       : list containing parameter C
    :param X_train : training features
    :param Y_train : training labels
    :return None
    """
    #for c in C:
    param = '-s 2 -c ' + str(c)
    model = lu.train(Y_train, X_train, param)
    lu.save_model("model/lmods2_"+str(c)+".model", model)

def test(c, Y_test, X_test):
    """
    This function takes in the test labels and features and prints out the accuracy
    :param C       :list containing parameter C
    :param X_train : test features
    :param Y_train : test labels
    :return None
    """
    #for c in C:
    model = lu.load_model("model/lmods2_"+str(c)+".model")
    p_letters, p_acc, p_val = lu.predict(Y_test, X_test, model)

    return p_letters, p_acc


def get_word_accuracy(orig_words, pred_words):
    """
    This function calculates the word wise accuracy
    :param orig_words: list containing correct labels
    :param pred_words: list containing predicted labels
    :return percentage of correct words predicted
    :rtype : float"""

    true_count = 0.0
    for w1,w2 in zip(orig_words, pred_words):
        if w1==w2:
            true_count += 1

    return true_count*100/len(orig_words)


def main():
    """ Execution begins here """
    train_file = "../../data/" + "train.txt"
    test_file  = "../../data/" + "test.txt"
    #C = [1, 10, 50, 100, 500, 1000, 5000]
    C = 5000

    # # Train
    # X_train, Y_train, wi_train = get_X_Y(train_file)
    # train(C, Y_train, X_train)

    # Test
    X_test,  Y_test, wi_test  = get_X_Y_wi(test_file)
    p_letters, p_acc = test(C, Y_test, X_test)

    orig_words, pred_words = get_words(Y_test, p_letters, wi_test)
    print("C = {} Word wise accuracy  : {} %".format(C, get_word_accuracy(orig_words, pred_words)))




if __name__ == "__main__": main()
## Letter wise Accuracy
# Without parameter -s 2
# Accuracy = 70.013% (18342/26198) (classification)
# Accuracy = 69.6847% (18256/26198) (classification)
# Accuracy = 61.4742% (16105/26198) (classification)
# Accuracy = 53.5423% (14027/26198) (classification)
# Accuracy = 49.8588% (13062/26198) (classification)
# Accuracy = 53.8362% (14104/26198) (classification)
# Accuracy = 48.435% (12689/26198) (classification)

# With parameter -s 2
# Accuracy = 69.9557% (18327/26198) (classification)
# Accuracy = 69.9061% (18314/26198) (classification)
# Accuracy = 69.9137% (18316/26198) (classification)
# Accuracy = 69.9328% (18320/26198) (classification)
# Accuracy = 69.9061% (18314/26198) (classification)
# Accuracy = 69.9328% (18320/26198) (classification)
# Accuracy = 69.9366% (18322/26198) (classification)

## Word wise Accuracy
# With parameter -s 2
#C = 1 Word wise accuracy  : 17.4178540273 %
#C = 10 Word wise accuracy  : 17.2433847049 %
#C = 50 Word wise accuracy  : 17.2433847049 %
#C = 100 Word wise accuracy  : 17.2143064844 %
#C = 500 Word wise accuracy  : 17.185228264 %
#C = 1000 Word wise accuracy  : 17.3015411457 %
#C = 5000 Word wise accuracy  : 17.3015411457 %

