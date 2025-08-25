import scipy
import numpy as np


d = np.array([1, 1, 0, 0, 0])
M = scipy.sparse.lil_matrix((5, 10))
A = scipy.sparse.diags(d, offsets=0, shape=M.shape)
M += A
A = scipy.sparse.diags(d, offsets=5, shape=M.shape)
M += A
import matplotlib.pyplot as plt

plt.spy(M)
plt.xticks(np.arange(0, M.shape[1], 1))
plt.yticks(np.arange(0, M.shape[0], 1))
plt.show()