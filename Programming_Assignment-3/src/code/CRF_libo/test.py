__author__ = "Libo Yin"
__author__ = "Manab Chetia"
# Q2.b

"""
This script uses learnt w and T matrix to predict the letters on the test data, and writes the result to prediction.txt.
"""
from lib import *
import numpy as np
import time

if __name__ == "__main__":
    w = np.loadtxt("result/learnt_w.txt", dtype=np.float64)
    t = np.loadtxt("result/learnt_t.txt", dtype=np.float64)
    data = load_data("data/test.txt")
    predictions = []
    correct_letter, correct_word = 0, 0
    start_time = time.time()
    for x, y in data:
        infer = max_sum(x, w, t).tolist()
        for i in range(0, len(y)):
            if infer[i] == y[i]:
                correct_letter += 1
        if infer == y.tolist():
            correct_word += 1
        predictions += infer
    print("testing time={}".format(time.time() - start_time))
    print("letter-wise accuracy={}%".format(100 * correct_letter / len(predictions)))
    print("word-wise accuracy={}%".format(100 * correct_word / len(data)))
    with open("result/prediction.txt", mode="w") as f:
        for i in predictions:
            f.write(str(i) + "\n")

# C=1000
# letter-wise accuracy=83.75066798992289%
# word-wise accuracy=47.223029950567025%
