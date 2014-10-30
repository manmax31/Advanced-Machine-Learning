__author__ = "Libo Yin"
__author__ = "Manab Chetia"

"""
Q 2a
This script calculates the log probability and its gradient.
"""

from math import exp, log
import numpy as np


def load_parameters(path, p):  # refers to: p
    """
    This function loads the required data from file model.txt
    :param  path: path of the file model.txt
    :param  p:    features in a letter i.e. 128
    :return w:    weights
    :return t:    transitions
    :rtype
    """
    raw = np.loadtxt(path + "model.txt")
    w   = np.array(raw[:26 * p]).reshape((26, p))
    t   = np.transpose(np.array(raw[26 * p:]).reshape((26, 26)))
    return w, t


def load_data(path):
    """
    This function gets the label and features from the train file
    :param path: path of train.txt file
    :return data: list of tuples containing (x,y)
    :rtype list of tuples
    """
    with open(path + "train.txt") as f:
        lines = [line.split() for line in f]
    data, x, y = [], [], []
    for l in lines:
        x.append( list(map(int, l[5:])) )
        y.append(ord(l[1]) - 97)
        if l[2] == "-1":
            data.append( ( np.array(x, dtype=np.bool), np.array(y, dtype=np.int) ) )
            x, y = [], []
    return data


def forward_dp(w, t, x):
    """
    This function for all possible paths from the first letter forwards, calculates the sum of partial, un-normalized probabilities
    :param w: weights
    :param t: transitions
    :param x: features
    :return dp: sum of partial, un-normalized probabilities
    :rtype : numpy array
    """
    m  = len(x)
    dp = np.zeros((m, 26), dtype=np.float64)

    for i in range(0, 26):  # the first line of the dp table
        dp[0, i] = exp(np.dot(w[i], x[0]))

    for i in range(1, m):                 # for all following lines of the dp table
        for j in range(0, 26):            # for each possible previous letter
            dp[i, j] = sum(dp[i-1, k] * exp(t[k, j]) for k in range(0, 26)) * exp(np.dot(w[j], x[i]))
                                          # here we rewrite the exp(sum) as prod(exp)
    return dp


def backward_dp(w, t, x):
    """
    This function for all possible paths from the last letter backwards, calculate the sum of partial, un-normalized probabilities
    :param w: weights
    :param t: transitions
    :param x: features
    :return dp: sum of partial, un-normalized probabilities
    :rtype : numpy array
    """
    m = len(x)
    dp = np.zeros((m, 26), dtype=np.float64)
    for i in range(0, 26):                    # the last line of the dp table
        dp[-1, i] = exp(np.dot(w[i], x[-1]))
    for i in range(m-2, -1, -1):              # for all previous lines of the dp table
        for j in range(0, 26):                # for each possible next letter
            dp[i, j] = sum(dp[i+1, k] * exp(t[j, k]) for k in range(0, 26)) * exp(np.dot(w[j], x[i]))
                                              # here we rewrite the exp(sum) as prod(exp)

    return dp


def gradient(w, t, x, y, p):
    """
    This function calculates the gradient with respect to W and T
    :param w: weights
    :param t: transitions
    :param x: features
    :param y: labels
    :param p: number of features for each letter
    :return nabla_w: gradients with respect to W
    :return nabla_t: gradients with respect to T
    :rtype : numpy array, numpy array
    """
    m        = len(x)
    forward  = forward_dp(w, t, x)
    backward = backward_dp(w, t, x)
    z        = sum(forward[-1])                               # partition function

    nabla_w  = np.empty((26, p), dtype=np.float64)
    nabla_t  = np.zeros((26, 26), dtype=np.float64)

    for i in range(0, 26):                                    # for each letter to take gradient against
        prob = x[0] * backward[0, i] + x[-1] * forward[-1, i]
        # sum of partial, unnormalized probability of all paths who contain letter @i on any bit
        for j in range(1, m-1):
            prob += x[j] * (forward[j, i] * backward[j, i] / exp(np.dot(w[i], x[j])))
        nabla_w[i] = sum(x[j] for j in range(0, m) if y[j] == i) - prob / z

    for i in range(1, m):                                     # for each transition in all possible paths
        for j in range(0, 26):
            for k in range(0, 26):
                nabla_t[j, k] -= forward[i-1, j] * backward[i, k] * exp(t[j, k])
                # sum of partial, unnormalized probability of all paths who contain transition <@j, @k> on any bit
    nabla_t /= z

    for i in range(1, m):
        nabla_t[y[i-1], y[i]] += 1

    return nabla_w, nabla_t


def log_probability(w, t, x, y):
    """
    This function calculates the log probability
    :param w: weights
    :param t: transitions
    :param x: features
    :param y: labels
    :return the log probability
    :rtype: float
    """
    m     = len(x)
    prob  = sum(np.dot(w[y[i]], x[i]) for i in range(0, m))
    prob += sum(t[y[i], y[i+1]] for i in range(0, m-1))
    z = sum(forward_dp(w, t, x)[-1])
    return log(exp(prob) / z)


def main():
    """ Execution begins here """
    path    = "../../data/"
    results = "../../results/"
    p       = 128
    w, t = load_parameters(path, p)

    # Storage for log probability and gradient
    log_prob, nablas_w, nablas_t = [], [], []

    print("Calculation of log probability and its gradient in progress...")
    for x, y in load_data(path):
        log_prob.append(log_probability(w, t, x, y))
        nw, nt = gradient(w, t, x, y, p)
        nablas_w.append(nw)  # 26 * @p
        nablas_t.append(nt)  # 26 * 26

    print("Log Probability: \n{}".format(log_prob))

    with open(results + "gradient.txt", mode="w") as f:
        for i in np.reshape(sum(nablas_w) / len(nablas_w), 26 * p):
            f.write(str(i) + "\n")
        for i in np.reshape(sum(nablas_t) / len(nablas_t), 26 * 26):
            f.write(str(i) + "\n")
        print("\nSuccessfully created gradient.txt ")


if __name__ == "__main__": main()
