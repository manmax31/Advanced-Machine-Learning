__author__ = "Libo Yin"
__author__ = "Manab Chetia"
"""
Q 4.
This script tampers the features of a word, then trains on the tampered features and finally prints the accuracy on test data set
"""

from scipy.misc import imrotate
import liblinearutil as lu
import numpy as np
import string
import copy


def get_X_Y(file):
    """
    This function extract the Y, X from the datasets
    :param file: name of the file
    :return X: features
    :return Y: labels
    :rtype : list, list
    """
    str_labels = np.loadtxt( file, usecols=[1], dtype=str)
    X          = np.loadtxt( file, usecols=range(5, 128+5)).tolist()
    Y          = get_int_labels( str_labels )
    return X, Y


def rotation(X, alpha):
    """
    This function rotates the image
    :param X    : vector representing the word
    :param alpha: angles by which image is to be rotated
    :return Y   : vector representing rotated word
    :rtype : list
    """
    Y = imrotate(X, alpha)
    len_x1, len_x2 = X.shape
    len_y1, len_y2 = Y.shape

    from_x = np.floor((len_y1 + 1 - len_x1)/2)
    from_y = np.floor((len_y2 + 1 - len_x2)/2)

    idx = np.where(Y==0)
    Y[idx] = X[idx]

    return Y


def translation(X, offset):
    """
    This function translates the image
    :param X     : vector representing the word
    :param offset: list representing translation offset
    :return Y    : vector representing translated word
    :rtype : list
    """
    Y = X
    len_x, len_y = X.shape
    ox = offset[0]
    oy = offset[1]

    if ox > 0 and oy > 0:
        # # Special case where ox and oy are both positive (used in this project)
        Y[1+ox:len_x, 1+oy:len_y] = X[1:len_x-ox, 1:len_y-oy]
    else:
        # # General case where ox and oy can be negative
        Y[ max(1,1+ox):min(len_x, len_x+ox), max(1,1+oy):min(len_y, len_y+oy) ] = X[max(1,1-ox):min(len_x, len_x-ox), max(1,1-oy):min(len_y, len_y-oy)]

    return Y


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


def get_words_and_pixvalues(file_name):
    """
    This function takes in a file and returns the a list containing the
    [[letters of word1],[letters of word2], ...] and [[pixel values of word1],[pixel values of word2],...]
    :param  file_name  : name of the file
    :return words      : list of words e.g. [[letters of word1],[letters of word2], ...]
    :return pixvalues  : list containing features of words e.g. [[[f1] [f2] [f3]], [[f4] [f5] [f6]]]
    :rtype : list, list
    """
    labels__next_words = np.loadtxt( file_name, usecols=[1,2], dtype=str)
    values_all         = np.loadtxt( file_name, usecols=range(5,128+5))

    str_labels = labels__next_words[:,0]
    next_words = labels__next_words[:,1]

    wi = np.insert(np.where(next_words=="-1"), 0, 0)

    int_labels = get_int_labels( str_labels )

    words        = []
    pixel_values = []

    sI = 0
    for i in xrange(0, len(wi) -1):
        eI  = wi[i + 1]

        word   = int_labels[sI:eI + 1]
        values = values_all[sI:eI + 1]

        sI = eI + 1

        words.append(word)
        pixel_values.append(values)

    return words, pixel_values


def tamper( commands, pixel_values, x_lines ):
    """
    This function reads the commands from transform.txt, takes in the pixelvalues and the number of lines to read from transform.txt
    and then tampers the pixels based on the commands 'r' or 't'
    :param commands     : contents of the transform.txt
    :param pixel_values : list containing features of words e.g. [[[f1] [f2] [f3]], [[f4] [f5] [f6]]]
    :param x_lines      : number of lines to read from transform.txt
    :rtype : list
    """
    pixel_values_tamper = copy.deepcopy(pixel_values)
    count = 0
    for command in commands:
        comm_arr = command.strip().split(" ")
        word_index = int(comm_arr[1]) - 1
        X_word = pixel_values[word_index]

        count += 1
        if count > x_lines:
            break

        if comm_arr[0] == 'r':
            rot_angle = float(comm_arr[2])
            X_word_rot = []
            for x_vec_letter in X_word:
                x_matrix_letter = x_vec_letter.reshape((8, 16))
                x_matrix_trans = rotation(x_matrix_letter, rot_angle)
                x_vec_letter_rot = x_matrix_trans.reshape((128,))
                X_word_rot.append(np.asarray(x_vec_letter_rot))
            pixel_values_tamper[word_index] = X_word_rot

        if comm_arr[0] == 't':
            offset = [int(comm_arr[2]), int(comm_arr[3])]
            X_word_rot = []
            for x_vec_letter in X_word:
                x_matrix_letter = x_vec_letter.reshape((8, 16))
                x_matrix_trans = translation(x_matrix_letter, offset)
                x_vec_letter_rot = x_matrix_trans.reshape((128,))
                X_word_rot.append(np.asarray(x_vec_letter_rot))
            pixel_values_tamper[word_index] = X_word_rot


    return pixel_values_tamper

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
        lu.save_model("model/lmods2_tamper"+str(c)+".model", model)


def test(C, Y_test, X_test):
    """
    This function takes in the test labels and features and prints out the accuracy
    :param C      : list containing parameter C
    :param X_test : test features
    :param Y_test : test labels
    :return None
    """
    for c in C:
        model = lu.load_model("model/lmods2_tamper"+str(c)+".model")
        p_letters, p_acc, p_val = lu.predict(Y_test, X_test, model)


def word_to_letters( words, tampered_pixels ):
    """
    This function reads in individual letters of the words and saves them in a list so that liblinear can understand it
    words   = [[l1 l2 l3], [l4 l5 l6], ...]
    letters = [l1 l2 l3 l4 l4 l6]. Here l1-l6 represents both labels and pixel_vector
    :param words           : list of words e.g. [[letters of word1],[letters of word2], ...]
    :param tampered_pixels : list containing features of words e.g. [[[f1] [f2] [f3]], [[f4] [f5] [f6]]]
    :return X_train : a list containing the features of each letter (which is formatted for liblinearutil)
    :return Y_train : a list containing the labels
    :rtype : list, list
    """
    Y_train = []
    X_train = []
    for word, value in zip(words, tampered_pixels):
        for letter, pixel in zip(word, value):
            Y_train.append(letter)
            X_train.append(pixel.tolist())
    return X_train, Y_train


def main():
    """ Execution begins here """
    path         = "../../data/"
    train_file   = path + "train.txt"
    test_file    = path + "test.txt"
    command_file = path + "transform.txt"

    C        = [100]
    x_lines  = 2000    # [0, 500, 1000, 1500, 2000] # Number of lines to read from transform.txt
    commands = open(command_file).readlines()

    words, train_words_values = get_words_and_pixvalues(train_file)

    print("Tampering in progress...")
    tampered_pixels           = tamper(commands, train_words_values, x_lines)
    print("Pixels are tampered successfully!")

    X_train, Y_train          = word_to_letters(words, tampered_pixels)

    # Train
    train(C, Y_train, X_train)

    # Test
    X_test,  Y_test  = get_X_Y(test_file)
    test(C, Y_test, X_test)


if __name__ == "__main__": main()
#0    lines tampered : Accuracy = 69.9557% (18327/26198) (classification)
#500  lines tampered : Accuracy = 36.4188% (9540/26198) (classification)
#1000 lines tampered : Accuracy = 29.7771% (7801/26198) (classification)
#1500 lines tampered : Accuracy = 25.1737% (6595/26198) (classification)
#2000 lines tampered : Accuracy = 24.9027% (6524/26198) (classification)