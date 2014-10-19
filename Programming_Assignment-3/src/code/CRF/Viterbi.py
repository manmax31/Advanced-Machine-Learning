import numpy as np

def formatData(raw_data):
    m         = 100
    n_letters = 26
    vec_len   = 128

    X = np.array    (           raw_data[:(m*vec_len)]).reshape((m, vec_len)                                              )
    W = np.array    (           raw_data[(m*vec_len):(m*vec_len+n_letters*vec_len)]).reshape((n_letters, vec_len)         )
    T = np.transpose( np.array( raw_data[(m*vec_len+n_letters*vec_len):]).reshape((n_letters, n_letters)  )               )

    return X, W, T

def getMaxStates(X, W, T):
    maxStates  = {}
    for x_row, li in zip(X, xrange(1, X.shape[0])):
        maxStates[li] = []
        for t_row in T:
            values = []
            for w_row, t_ij in zip(W, t_row):
                values.append( np.dot(x_row, w_row) + t_ij )

            maxIndex = np.argmax(values)
            maxStates[li].append(maxIndex)
    return maxStates

def getPaths(maxStates, W, X):
    paths = {}
    for e,i in zip(maxStates[99], xrange(W.shape[0])):
        paths[i]=[e]

    for pi in paths:
        path = paths[pi]
        for index in xrange(X.shape[0]-1, 0, -1):
            path.append(maxStates[index][path[-1]])

    return paths


def main():
    path = "../../data/"
    data = np.loadtxt(path + "decode_input.txt")

    X, W, T = formatData(data)

    maxStates =  getMaxStates(X, W, T)
    paths     =  getPaths(maxStates, W, X)

    print len(paths)


    #print maxStates[99]
    # paths = {}
    # for e,i in zip(maxStates[99], xrange(W.shape[0])):
    #     paths[i]=[e]
    # #print paths
    #
    # # ini = maxStates[99][0]
    # # path = []
    # # path.append(ini)
    # # for index in xrange(X.shape[0]-1, 0, -1):
    # #     path.append(maxStates[index][path[-1]])
    # #print path
    #
    # for pi in paths:
    #     path = paths[pi]
    #     for index in xrange(X.shape[0]-1, 0, -1):
    #         path.append(maxStates[index][path[-1]])













if __name__ == "__main__":main()