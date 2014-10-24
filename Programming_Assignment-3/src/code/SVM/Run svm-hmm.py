__author__ = 'manabchetia'

import subprocess as sp

def train_svm(c, train_path, model_path):
    sp.call(["svm_hmm_learn", "-c", str(c), train_path, model_path])

def test_svm(test_path, model_path, outtag_path):
    sp.call(["svm_hmm_classify", test_path, model_path, outtag_path])

def main():
    c = 500
    train_path  = '../../data/train_struct.txt'
    test_path   = '../../data/test_struct.txt'
    model_path  = 'model/mod' + str(c) + '.model'
    outtag_path = 'outtags/test' + str(c) + '.outtags'

    train_svm(c, train_path, model_path)
    test_svm(test_path, model_path, outtag_path)


if __name__ == "__main__": main()
