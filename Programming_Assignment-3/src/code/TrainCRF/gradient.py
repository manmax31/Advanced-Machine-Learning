# Q2.a
"""
This script contains functions used by other scripts.
"""

from lib import *
import numpy as np


def load_parameters():  # refers to: p
    raw = np.loadtxt("data/model.txt")
    w = np.array(raw[:26 * p]).reshape((26, p))
    t = np.transpose(np.array(raw[26 * p:]).reshape((26, 26)))
    return w, t

p = 128
w, t = load_parameters()
if __name__ == "__main__":
    nablas_w, nablas_t, log_probs = [], [], []
    for x, y in load_data("data/train.txt"):
        nw, nt = gradient(w, t, x, y)
        nablas_w.append(nw)  # 26 * @p
        nablas_t.append(nt)  # 26 * 26
        log_probs.append(log_probability(w, t, x, y))
    nabla_w = sum(nablas_w) / len(nablas_w)
    nabla_t = sum(nablas_t) / len(nablas_t)
    print(sum(log_probs) / len(log_probs))
    np.savetxt("result/nabla_w.txt", nabla_w)
    np.savetxt("result/nabla_t.txt", nabla_t)
    with open("result/gradient.txt", mode="w") as f:
        for i in np.reshape(nabla_w, 26 * p):
            f.write(str(i) + "\n")
        for i in np.reshape(np.transpose(nabla_t), 26 * 26):
            f.write(str(i) + "\n")

# average log probability=-31.288437439649147
