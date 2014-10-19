import numpy as np

'''
This function formats the data into X, W, T from the given file
'''
def formatData(raw_data):
    m         = 100
    n_letters = 26
    vec_len   = 128

    X = np.array    (           raw_data[:(m*vec_len)]).reshape((m, vec_len)                                              )
    W = np.array    (           raw_data[(m*vec_len):(m*vec_len+n_letters*vec_len)]).reshape((n_letters, vec_len)         )
    T = np.transpose( np.array( raw_data[(m*vec_len+n_letters*vec_len):]).reshape((n_letters, n_letters)  )               )

    return X, W, T

'''
This function gets the most likely transition state for a state from all possible states in the previous layer
'''
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

'''
This function gets all the 26 possible paths
'''
def getPaths(maxStates, W, X):
    paths = {}
    for e,i in zip(maxStates[len(maxStates)], xrange(W.shape[0])):
        paths[i]=[e]

    for pi in paths:
        path = paths[pi]
        for index in xrange(X.shape[0]-1, 0, -1):
            path.append(maxStates[index][path[-1]])

    return paths

'''
This function gets the most likely state based on all 26 possible states
'''
def getMostLikelyPath(paths, T):
    Tp = np.transpose(T)

    scores = []
    for pi in paths:
        path = paths[pi]
        score = 0.0
        for i in xrange(len(path)-1):
            score += Tp[path[i]][path[i+1]]
        scores.append(score)

    maxPathIndex =  np.argmax(scores)
    mostLikelyPath = paths[maxPathIndex]
    return mostLikelyPath

'''
This function write the states of the most likely path to the file
'''
def writeToFile(mostLikelyPath, filepath):
    f = open(filepath, 'w')
    for state in mostLikelyPath:
        f.write(str(state) + "\n")
    f.close()
    print "Successfully created decode_output.txt "

'''MAIN method'''
def main():
    path = "../../data/"
    data = np.loadtxt(path + "decode_input.txt")

    X, W, T = formatData(data)

    maxStates      =  getMaxStates(X, W, T)
    paths          =  getPaths(maxStates, W, X)
    mostLikelyPath =  getMostLikelyPath(paths, T)
    writeToFile(mostLikelyPath, "../../results/decode_output.txt")


if __name__ == "__main__":main()