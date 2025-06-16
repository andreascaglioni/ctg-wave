import numpy as np

t = np.linspace(0, 1, 3)
x = np.linspace(0, 1, 11).reshape((-1, 1))  # always 2D. nth ROW denotes n-th dof; # column = dim(D)
rep_v = np.ones((t.shape[0], 1))


long_x = np.kron(x, rep_v)
print(long_x)
