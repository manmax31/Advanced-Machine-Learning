import numpy as np
import os.path
import copy

basepath = os.path.dirname(__file__)
decode_input_filepath = os.path.abspath(os.path.join(basepath, "..", "data", "decode_input.txt"))
#train_filepath = os.path.abspath(os.path.join(basepath, "..", "data", "train.txt"))
#train_struct_filepath = os.path.abspath(os.path.join(basepath, "..", "data", "train_struct.txt"))


decode_input = np.loadtxt(open(decode_input_filepath,"rb"))


x_dash = np.array(decode_input[:12800]).reshape((100, 128))
w_dash = np.array(decode_input[12800:16128]).reshape((26, 128))
T_dash = np.array(decode_input[16128:]).reshape((26, 26))

# decode_ input
# 128 for each x, 100 tota
# Each of the 26 letters is represented by 1x128 weights
#[128] x 100
#[128] x 26
# 26 x 26 transition