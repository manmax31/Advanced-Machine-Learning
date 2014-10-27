__author__ = "Libo Yin"
"""
Q 1c. Max-Sum decoder
This script implements max-sum algorithm to decode the file decode_input.txt
"""
import itertools as it
import numpy as np


def get_parameters(raw_data, m, p):
    """
    This function gets the parameters X, W, T from the file decode_input
    :param raw_data: filename
    :param m: number of letters in the word i.e. 100
    :param p: number of pixel values for each letter i.e 128
    :return X: matrix of pixel values
    :return W: matrix of weight values
    :return T: transition matrix
    :rtype: list, list, list
    """
    X = np.array(raw_data[:m * p]).reshape((m, p))
    W = np.array(raw_data[m * p:m * p + 26 * p]).reshape((26, p))
    T = np.transpose(np.array(raw_data[(m * p + 26 * p):]).reshape((26, 26)))
    return X, W, T


def write_to_file(letters, filename):
    """
    This function writes the decoded_output to a file decode_output.txt
    :param letters: list containing the decoded output
    :param filename: decode_output.txt
    :return: None
    """
    with open(filename, mode="w") as f:
        for l in letters:
            f.write(str(l + 1) + "\n")  # +1 since python index starts from 0
    print("\nSuccessfully created decode_output.txt ")


def max_sum(m, X, W, T):
    """
    This function runs the Max-Sum algorithm to get the most likely states
    :param m: no of letters in the word i.e. 100
    :param X: matrix of pixel values
    :param W: matrix of weight values
    :param T: transition matrix
    :return decoded_output: contains the decoded output
    :rtype list
    """
    dp_argmax = np.zeros((m, 26), dtype=np.int)      # optimal pointers
    dp_vmax   = np.zeros((m, 26), dtype=np.float64)  # max values corresponding to the pointers

    for i in range(0, 26):                           # first row of the dp table
        dp_vmax[0, i] = np.dot(W[i], X[0])

    for i in range(1, m):                            # for all rows of the dp table
        for j in range(0, 26):                       # for each current letter
            prev = np.copy(dp_vmax[i - 1])           # the previous row of the dp table
            for k in range(0, 26):                   # for each previous letter
                prev[k] += T[k, j]                   # the dot product shall be added later since it is a constant wrt argmax_k
            k_max           = np.argmax(prev)
            dp_argmax[i, j] = k_max                  # pointer to the previous link. note that the first row in @dp_argmax is empty
            dp_vmax[i, j]   = prev[k_max] + np.dot(W[j], X[i])  # add the dot product here

    decoded_output      = np.zeros(m, dtype=np.int)
    decoded_output[m-1] = np.argmax(dp_vmax[m-1])               # the last choice depends on the value

    for i in range(m-1, 0, -1):
        decoded_output[i-1] = dp_argmax[i][decoded_output[i]]

    return decoded_output.tolist()


def brute_force(X, W, T):
    """
    This function runs the brute force algorithm to get the most likely states on 5 letters
    :param X: matrix of pixel values
    :param W: matrix of weight values
    :param T: transition matrix
    :return: maximum configuration
    :rtype: list
    """
    m = 5                                             # infer the first @m letters of @x
    ys = list(it.product(range(26), repeat=m))
    scores = []
    for y in ys:
        s = 0
        for i in range(0, m - 1):
            s += np.dot(W[y[i]], X[i]) + T[y[i]][y[i+1]]
        s += np.dot(W[y[m-1]], X[m-1])
        scores.append(s)
    y_max = max(zip(ys, scores), key=lambda x: x[1])[0]
    return y_max


def main():
    """Execution begins here"""
    
    path     = "../../data/"
    raw_data = np.loadtxt(path + "decode_input.txt")

    m = 100                                           # number of letters in a word
    p = 128                                           # number of pixels in a letter
    X, W, T = get_parameters(raw_data, m, p)

    decode_output = [x + 1 for x in max_sum(m, X, W, T)]
    print("Decoded Output:")
    print(decode_output)
    write_to_file(decode_output, "../../results/decode_output.txt")

if __name__ == "__main__": main()
