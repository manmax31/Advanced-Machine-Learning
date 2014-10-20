__author__ = 'manab chetia'
"""
Q 1c. Max-Sum
This script implements max-sum algorithm to decode the file decode_input.txt
"""

import numpy as np


def format_data(raw_data):
    """ This function formats the data into X, W, T from the given file 
    :rtype : numpy arrays
    """
    m         = 100
    n_letters = 26
    vec_len   = 128

    X = np.array(raw_data[:(m * vec_len)]).reshape((m, vec_len))
    W = np.array(raw_data[(m * vec_len):(m * vec_len + n_letters * vec_len)]).reshape((n_letters, vec_len))
    T = np.transpose(np.array(raw_data[(m * vec_len + n_letters * vec_len):]).reshape((n_letters, n_letters)))

    return X, W, T


def get_max_states(X, W, T):
    """ This function gets the most likely transition state for a state from all possible states in the previous layer 
    :rtype : dictionary
    """
    max_states = {}
    for x_row, li in zip(X, xrange(1, X.shape[0])):
        max_states[li] = []
        for t_row in T:
            values = []
            for w_row, t_ij in zip(W, t_row):
                values.append(np.dot(x_row, w_row) + t_ij)

            max_index = np.argmax(values)
            max_states[li].append(max_index)
    return max_states


def get_paths(max_states, W, X):
    """ This function gets all the 26 possible paths
    :rtype : dictionary
    """
    paths = {}
    for e, i in zip(max_states[len(max_states)], xrange(W.shape[0])):
        paths[i] = [e]

    for pi in paths:
        path = paths[pi]
        for index in xrange(X.shape[0] - 1, 0, -1):
            path.append(max_states[index][path[-1]])

    return paths


def get_most_likely_path(paths, T):
    """ This function gets the most likely state based on all 26 possible states
    :rtype : list
    """
    Tp = np.transpose(T)

    scores = []
    for pi in paths:
        path = paths[pi]
        score = 0.0
        for i in xrange(len(path) - 1):
            score += Tp[path[i]][path[i + 1]]
        scores.append(score)

    # print("Score of max path:{}".format(max(scores)))
    max_path_index = np.argmax(scores)
    most_likely_path = paths[max_path_index]
    most_likely_path.reverse()
    return most_likely_path


def write_to_file(most_likely_path, file_path):
    """ This function write the states of the most likely path to the file """
    f = open(file_path, 'w')
    for state in most_likely_path:
        f.write(str(state + 1) + "\n")
    f.close()
    print "Successfully created decode_output.txt "


def main():
    """ MAIN method : Execution starts here """
    path     = "../../data/"
    raw_data = np.loadtxt(path + "decode_input.txt")

    X, W, T          = format_data(raw_data)
    max_states       = get_max_states(X, W, T)
    paths            = get_paths(max_states, W, X)
    most_likely_path = get_most_likely_path(paths, T)
    write_to_file(most_likely_path, "../../results/decode_output.txt")


if __name__ == "__main__": main()