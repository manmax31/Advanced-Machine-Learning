# Q2.b
from lib import *
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time


def objective(theta):
    w = np.reshape(theta[:26 * p], (26, p))
    t = np.reshape(theta[26 * p:], (26, 26))
    score = -c * sum(log_probability(w, t, x, y) for x, y in data) / len(data)
    score += sum((np.linalg.norm(x)) ** 2 for x in w) / 2
    score += sum(sum(x ** 2 for x in row) for row in t) / 2
    return score


def obj_prime(theta):
    w = np.reshape(theta[:26 * p], (26, p))
    t = np.reshape(theta[26 * p:], (26, 26))
    nablas_w, nablas_t = [], []
    for x, y in data:
        nw, nt = gradient(w, t, x, y)
        nablas_w.append(nw)
        nablas_t.append(nt)
    nabla_w = -c * sum(nablas_w) / len(data) + w
    nabla_t = -c * sum(nablas_t) / len(data) + t
    return np.concatenate((np.reshape(nabla_w, 26 * p), np.reshape(nabla_t, 26 ** 2)))


p = 128
c = 1000
if __name__ == "__main__":
    data = load_data("data/train.txt")
    start_time = time.time()
    theta, min_obj, _ = fmin_l_bfgs_b(func=objective, x0=np.zeros(26 * p + 26 ** 2), fprime=obj_prime)
    print("learning time={}".format(time.time() - start_time))
    print("c={}".format(c))
    print("value={}".format(min_obj))
    w = np.reshape(theta[:26 * p], (26, p))
    t = np.reshape(theta[26 * p:], (26, 26))
    np.savetxt("result/learnt_w.txt", w)
    np.savetxt("result/learnt_t.txt", t)
    with open("result/solution.txt", mode="w") as f:
        for i in np.reshape(w, 26 * p):
            f.write(str(i) + "\n")
        for i in np.reshape(np.transpose(t), 26 * 26):
            f.write(str(i) + "\n")
