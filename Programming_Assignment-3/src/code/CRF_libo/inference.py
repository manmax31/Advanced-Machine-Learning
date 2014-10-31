# Q1.c
from lib import *
import numpy as np


def load_parameters():  # refers to: m, p
    raw = np.loadtxt("data/decode_input.txt")
    x = np.array(raw[:m * p]).reshape((m, p))
    w = np.array(raw[m * p:m * p + 26 * p]).reshape((26, p))
    t = np.transpose(np.array(raw[(m * p + 26 * p):]).reshape((26, 26)))
    return x, w, t

m = 100  # number of letters in the word
p = 128  # number of pixels in a letter
if __name__ == "__main__":
    x, w, t = load_parameters()
    letters = [i + 1 for i in max_sum(x, w, t)]  # +1 since python index starts from 0
    print(letters)
    with open("result/decode_output.txt", mode="w") as f:
        for l in letters:
            f.write(str(l) + "\n")

# print([i + 1 for i in brute_force(x, w, t, 5)])
