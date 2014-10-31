__author__ = "Libo Yin"
__author__ = "Manab Chetia"
"""
This script contains functions used by other scripts.
"""


from math import exp, log
import itertools as it
import numpy as np


def max_sum(x, w, t):
    m = len(x)
    dp_argmax = np.zeros((m, 26), dtype=np.int)  # backward pointers
    dp_vmax = np.zeros((m, 26), dtype=np.float64)  # max values corresponding to the pointers
    for i in range(0, 26):  # first row of the dp table
        dp_vmax[0, i] = np.dot(w[i], x[0])
    for i in range(1, m):  # for all rows of the dp table
        for j in range(0, 26):  # for each current letter
            prev = np.copy(dp_vmax[i - 1])  # the previous row of the dp table
            for k in range(0, 26):  # for each previous letter
                prev[k] += t[k, j]  # the dot product shall be added later since it is a constant wrt argmax_k
            k_max = np.argmax(prev)
            dp_argmax[i, j] = k_max  # point to the previous link. note that @dp_argmax[0] is empty
            dp_vmax[i, j] = prev[k_max] + np.dot(w[j], x[i])  # the dot product is for the current letter
    word = np.zeros(m, dtype=np.int)  # backtrack the dp tables
    word[m-1] = np.argmax(dp_vmax[m-1])  # the last choice depends on the min_obj
    for i in range(m-1, 0, -1):  # all previous choices have been calculated
        word[i-1] = dp_argmax[i][word[i]]
    return word


def brute_force(x, w, t, m):  # infer the first @m letters of @x
    letters = list(it.product(range(26), repeat=m))
    scores = []
    for l in letters:
        scores.append(sum(np.dot(w[l[i]], x[i]) + t[l[i]][l[i+1]] for i in range(0, m - 1)) + np.dot(w[l[m-1]], x[m-1]))
    return max(zip(letters, scores), key=lambda x: x[1])[0]


def forward_dp(w, t, x):
    # for all possible paths from the first letter forwards, calculate the sum of partial, unormalized probabilities
    m = len(x)
    dp = np.zeros((m, 26), dtype=np.float64)
    for i in range(0, 26):  # the first line of the dp table
        dp[0, i] = exp(np.dot(w[i], x[0]))
    for i in range(1, m):  # for all following lines of the dp table
        for j in range(0, 26):  # for each possible previous letter
            dp[i, j] = sum(dp[i-1, k] * exp(t[k, j]) for k in range(0, 26)) * exp(np.dot(w[j], x[i]))
            # here we rewrite the exp(sum) as prod(exp)
    return dp


def backward_dp(w, t, x):
    # for all possible paths from the last letter backwards, calculate the sum of partial, unormalized probabilities
    m = len(x)
    dp = np.zeros((m, 26), dtype=np.float64)
    for i in range(0, 26):  # the last line of the dp table
        dp[-1, i] = exp(np.dot(w[i], x[-1]))
    for i in range(m-2, -1, -1):  # for all previous lines of the dp table
        for j in range(0, 26):  # for each possible next letter
            dp[i, j] = sum(dp[i+1, k] * exp(t[j, k]) for k in range(0, 26)) * exp(np.dot(w[j], x[i]))
            # here we rewrite the exp(sum) as prod(exp)
    return dp


def gradient(w, t, x, y):
    p = 128
    m = len(x)
    forward = forward_dp(w, t, x)
    backward = backward_dp(w, t, x)
    z = sum(forward[-1])  # partition function
    nabla_w = np.empty((26, p), dtype=np.float64)
    for i in range(0, 26):  # for each letter to take gradient against
        ws = x[0] * backward[0, i] + x[-1] * forward[-1, i]
        # weighted sum of pictures on partial, unnormalized probability of all paths who contain letter @i on any bit
        for j in range(1, m-1):
            ws += x[j] * (forward[j, i] * backward[j, i] / exp(np.dot(w[i], x[j])))
        nabla_w[i] = sum(x[j] for j in range(0, m) if y[j] == i) - ws / z
    nabla_t = np.zeros((26, 26), dtype=np.float64)
    for i in range(1, m):  # for each transition in all possible paths
        for j in range(0, 26):
            for k in range(0, 26):
                nabla_t[j, k] -= forward[i-1, j] * backward[i, k] * exp(t[j, k])
                # sum of partial, unnormalized probability of all paths who contain transition <@j, @k> on any bit
    nabla_t /= z
    for i in range(1, m):
        nabla_t[y[i-1], y[i]] += 1
    return nabla_w, nabla_t


def log_probability(w, t, x, y):
    psi = sum(np.dot(w[y[i]], x[i]) for i in range(0, len(x)))
    psi += sum(t[y[i], y[i+1]] for i in range(0, len(x)-1))
    z = sum(forward_dp(w, t, x)[-1])
    return log(exp(psi) / z)


def load_data(path):
    with open(path) as f:
        lines = [line.split() for line in f]
    data, x, y = [], [], []
    for l in lines:
        pixels = np.array(list(map(int, l[5:])), dtype=np.float64)
        if any(p > 1 for p in pixels):
            pixels /= 255
        x.append(pixels)
        y.append(ord(l[1]) - 97)
        if l[2] == "-1":
            data.append((np.array(x, dtype=np.float64), np.array(y, dtype=np.int)))
            x, y = [], []
    return data
