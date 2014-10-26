import itertools as it
import numpy as np


def load_parameters(raw):  # refers to: m, p
    X = np.array(raw[:m * p]).reshape((m, p))
    W = np.array(raw[m * p:m * p + 26 * p]).reshape((26, p))
    T = np.transpose(np.array(raw[(m * p + 26 * p):]).reshape((26, 26)))
    return X, W, T


def write_to_file(letters, filename):
    with open(filename, mode="w") as f:
        for l in letters:
            f.write(str(l + 1) + "\n")  # +1 since python index starts from 0


def max_sum():  # refers to: x, w, t
    dp_argmax = np.zeros((m, 26), dtype=np.int)  # optimal pointers
    dp_vmax = np.zeros((m, 26), dtype=np.float64)  # max values corresponding to the pointers
    for i in range(0, 26):  # first row of the dp table
        dp_vmax[0, i] = np.dot(w[i], x[0])
    for i in range(1, m):  # for all rows of the dp table
        for j in range(0, 26):  # for each current letter
            prev = np.copy(dp_vmax[i - 1])  # the previous row of the dp table
            for k in range(0, 26):  # for each previous letter
                prev[k] += t[k, j]  # the dot product shall be added later since it is a constant wrt argmax_k
            k_max = np.argmax(prev)
            dp_argmax[i, j] = k_max  # pointer to the previous link. note that the first row in @dp_argmax is empty
            dp_vmax[i, j] = prev[k_max] + np.dot(w[j], x[i])  # add the dot product here
    word = np.zeros(m, dtype=np.int)
    word[m-1] = np.argmax(dp_vmax[m-1])  # the last choice depends on the value
    for i in range(m-1, 0, -1):
        word[i-1] = dp_argmax[i][word[i]]
    return word.tolist()


def brute_force():  # refers to: x, w, t
    m = 5  # infer the first @m letters of @x
    ys = list(it.product(range(26), repeat=m))
    scores = []
    for y in ys:
        s = 0
        for i in range(0, m - 1):
            s += np.dot(w[y[i]], x[i]) + t[y[i]][y[i+1]]
        s += np.dot(w[y[m-1]], x[m-1])
        scores.append(s)
    y_max = max(zip(ys, scores), key=lambda x: x[1])[0]
    return y_max

m = 100  # number of letters in a word
p = 128  # number of pixels in a letter
x, w, t = load_parameters(np.loadtxt("../../data/decode_input.txt"))
# letters = [x + 1 for x in brute_force()]
# print(letters)
letters = [x + 1 for x in max_sum()]
print(letters)
write_to_file(letters, "../../results/decode_output_libo.txt")
