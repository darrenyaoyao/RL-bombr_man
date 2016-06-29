import numpy as np
X = np.load("features.npy")
for i in range(len(X)):
    print X[i]
print X.shape
