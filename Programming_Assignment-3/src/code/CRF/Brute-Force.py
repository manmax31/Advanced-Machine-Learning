__author__ = 'manabchetia'
"""
Q 1c. Brute Force decoder
This script implements brute force in 4 letters
"""
import numpy     as np
import itertools as it

def format_data(raw_data):
    """ This function formats the data into X, W, T from the given file """
    m         = 100
    n_letters = 26
    vec_len   = 128

    X = np.array( raw_data[:(m*vec_len)]).reshape((m, vec_len))
    W = np.array( raw_data[(m*vec_len):(m*vec_len+n_letters*vec_len)]).reshape((n_letters, vec_len))
    T = np.array( raw_data[(m*vec_len+n_letters*vec_len):]).reshape((n_letters, n_letters))

    return X, W, T

def brute_force(T):
    configurations = np.asarray(list(it.product(range(26), repeat=4)))
    scores = []
    for configuration in configurations:
        score = T[configuration[0]][configuration[1]] + T[configuration[1]][configuration[2]] + T[configuration[2]][configuration[3]]
        scores.append(score)
    max_path_index   = np.argmax(scores)
    most_likely_path = configurations[max_path_index]
    print most_likely_path

def main():
    path = "../../data/"
    data = np.loadtxt(path + "decode_input.txt")

    X, W, T = format_data(data)
    brute_force(T)

if __name__ == "__main__":main()