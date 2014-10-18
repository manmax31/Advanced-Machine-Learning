import numpy as np
import os.path
import itertools as it

basepath = os.path.dirname(__file__)
decode_input_filepath = os.path.abspath(os.path.join(basepath, "..", "data", "decode_input.txt"))
decode_input = np.loadtxt(open(decode_input_filepath,"rb"))

m = 100
m_brute_force = 3
n_letters = 26
vec_len = 128

x_word = np.array(decode_input[:(m*vec_len)]).reshape((m, vec_len))
w_word = np.array(decode_input[(m*vec_len):(m*vec_len+n_letters*vec_len)]).reshape((n_letters, vec_len))
T_word = np.array(decode_input[(m*vec_len+n_letters*vec_len):]).reshape((n_letters, n_letters))

configurations = np.asarray(list(it.product(range(n_letters), repeat=m_brute_force)))


brute_force_solution = 0

for y in configurations:
	p = 0
	p_max = 0
	for i in range(m_brute_force-1):
		p += T_word[y[i]][y[i+1]]
	if p >= p_max:
		p_max = p
		brute_force_solution = y

print brute_force_solution 
	
