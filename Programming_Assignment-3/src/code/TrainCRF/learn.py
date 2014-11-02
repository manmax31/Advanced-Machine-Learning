__author__ = "Libo Yin"
__author__ = "Manab Chetia"

"""
# Q2.b
This script learns w matrix and T matrix from the training data, and writes them to learnt_w.txt, learnt_t.txt, and solution.txt.
"""
from lib import *
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time


def objective(theta):
    """
    The objective function
    :param theta: parameters
    :return: score
    """
    W = np.reshape(theta[:26 * p], (26, p))
    T = np.reshape(theta[26 * p:], (26, 26))
    score = -c * sum(log_probability(W, T, x, y) for x, y in data) / len(data)
    score += sum((np.linalg.norm(x)) ** 2 for x in W) / 2
    score += sum(sum(x ** 2 for x in row) for row in T) / 2
    return score


def obj_prime(theta):
    """
    Gradient of objective function
    :param theta: parameters
    :return: gradient with respect to W and T
    """
    W = np.reshape(theta[:26 * p], (26, p))
    T = np.reshape(theta[26 * p:], (26, 26))
    nablas_W, nablas_T = [], []
    for x, y in data:
        nw, nt = gradient(W, T, x, y)
        nablas_W.append(nw)
        nablas_T.append(nt)
    nabla_w = -c * sum(nablas_W) / len(data) + W
    nabla_t = -c * sum(nablas_T) / len(data) + T
    return np.concatenate((np.reshape(nabla_w, 26 * p), np.reshape(nabla_t, 26 ** 2)))

# Global data
p = 128
c = 1
data_path    = "../../data/"
data = load_data(data_path + "train.txt")

def main():
    results = "../../results/"
    start_time = time.time()

    print("Training in progress...")
    theta, min_obj, _ = fmin_l_bfgs_b(func=objective, x0=np.zeros(26 * p + 26 ** 2), fprime=obj_prime)
    print("Learning Time: {}".format(time.time() - start_time))
    print("C = {}".format(c))
    print("Objective Value = {}".format(min_obj))

    w = np.reshape( theta[:26 * p], (26, p)  )
    t = np.reshape( theta[26 * p:], (26, 26) )
    np.savetxt( results+"learnt_w.txt", w )
    np.savetxt( results+"learnt_t.txt", t )

    print("Writing to solution.txt")
    with open(results+"solution"+str(c)+".txt", mode="w") as f:
        for i in np.reshape(w, 26 * p):
            f.write(str(i) + "\n")
        for i in np.reshape(np.transpose(t), 26 * 26):
            f.write(str(i) + "\n")
        print("solution.txt created successfully")

if __name__ == "__main__":main()
