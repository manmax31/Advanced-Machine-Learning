__author__ = 'manabchetia'

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imrotate

# plt.plot([1, 10, 50, 100, 500, 1000, 5000], [65.22, 75.04, 80.75, 82.34, 84.12, 84.75, 85.16])
# plt.xlabel("C")
# plt.ylabel("Accuracy in %")
# plt.xscale("log")
# plt.title("Letter Wise Accuracy using SVM-Struct")
# plt.style.use(['ggplot'])
# plt.show()


def rotation( X, alpha ):
    """
    This function rotates the image
    :param X    : vector representing the word
    :param alpha: angles by which image is to be rotated
    :return Y   : vector representing rotated word
    :rtype : list
    """
    Y = imrotate(X, alpha)
    len_x1, len_x2 = X.shape
    len_y1, len_y2 = Y.shape

    from_x = np.floor((len_y1 + 1 - len_x1) / 2)
    from_y = np.floor((len_y2 + 1 - len_x2) / 2)
    Y = Y[from_x:from_x+len_x1, from_y:from_y+len_x2]

    idx = np.where(Y == 0)
    Y[idx] = X[idx]

    return Y

x=np.arange(9).reshape(3,3)
print x
print rotation(x,15)