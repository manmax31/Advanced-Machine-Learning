__author__ = 'manabchetia'
"""
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


def get_X_Y(file):
    """
    This function extract the Y, X from the datasets
    :param file : training or test file
    :return X   : features
    :return Y   : labels
    :rtype : list, list
    """
    str_labels = np.loadtxt( file, usecols=[1], dtype=str)
    X          = np.loadtxt( file, usecols=range(5, 128+5)).tolist()
    Y          = get_int_labels( str_labels )
    return X, Y


def train(C, Y_train, X_train):
    """
    This function takes in the training labels and features and creates a model and saves that model
    :param C       : list containing parameter C
    :param X_train : training features
    :param Y_train : training labels
    :return None
    """
    for c in C:
        param = '-s 2 -c ' + str(c)
        model = lu.train(Y_train, X_train, param)
        lu.save_model("model/lmods2_"+str(c)+".model", model)

def test(C, Y_test, X_test):
    """
    This function takes in the test labels and features and prints out the accuracy
    :param C       :list containing parameter C
    :param X_train : test features
    :param Y_train : test labels
    :return None
    """
    for c in C:
        model = lu.load_model("model/lmods2_"+str(c)+".model")
        p_letters, p_acc, p_val = lu.predict(Y_test, X_test, model)

def main():
    """ Execution begins here """
    train_file = "../../data/" + "train.txt"
    test_file  = "../../data/" + "test.txt"
    C = [1, 10, 50, 100, 500, 1000, 5000]

    # Train
    X_train, Y_train = get_X_Y(train_file)
    train(C, Y_train, X_train)

    # Test
    X_test,  Y_test  = get_X_Y(test_file)
    test(C, Y_test, X_test)



if __name__ == "__main__": main()
## Without parameter -s 2
# Accuracy = 70.013% (18342/26198) (classification)
# Accuracy = 69.6847% (18256/26198) (classification)
# Accuracy = 61.4742% (16105/26198) (classification)
# Accuracy = 53.5423% (14027/26198) (classification)
# Accuracy = 49.8588% (13062/26198) (classification)
# Accuracy = 53.8362% (14104/26198) (classification)
# Accuracy = 48.435% (12689/26198) (classification)

## With parameter -s 2
# Accuracy = 69.9557% (18327/26198) (classification)
# Accuracy = 69.9061% (18314/26198) (classification)
# Accuracy = 69.9137% (18316/26198) (classification)
# Accuracy = 69.9328% (18320/26198) (classification)
# Accuracy = 69.9061% (18314/26198) (classification)
# Accuracy = 69.9328% (18320/26198) (classification)
# Accuracy = 69.9366% (18322/26198) (classification)
