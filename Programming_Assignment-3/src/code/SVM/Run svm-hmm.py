__author__ = 'manabchetia'
"""
This script runs the binary executables svm_hmm_learn to create a model and then runs svm_hmm_classify to generate a file
which stores the predicted labels.
"""

import subprocess as sp

def train_svm(c, train_path, model_path):
    """
    This function runs the executable svm_hmm_learn on the command line
    :param test_path:   path of the test file
    :param model_path:  path of the model file
    :return: None
    """
    sp.call(["svm_hmm_learn", "-c", str(c), train_path, model_path])


def test_svm(test_path, model_path, outtag_path):
    """
    This function runs the executable svm_hmm_classify on the command line
    :param test_path:   path of the test file
    :param model_path:  path of the model file
    :param outtag_path: path of the predicted file
    :return: None
    """
    sp.call(["svm_hmm_classify", test_path, model_path, outtag_path])


def main():
    """ Execution begins here """
    c = 1000 # [1, 10, 100, 1000]

    train_path  = '../../data/train_struct.txt'
    test_path   = '../../data/test_struct.txt'

    model_path  = 'model/mod'    + str(c) + '.model'
    outtag_path = 'outtags/test' + str(c) + '.outtags'

    train_svm(c, train_path, model_path)
    test_svm(test_path, model_path, outtag_path)


if __name__ == "__main__": main()
