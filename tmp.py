import matplotlib.pyplot as plt
from dolfinx import mesh
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_SELF
n_x = 9
msh = mesh.create_unit_interval(comm, n_x)
xx = msh.geometry.x[:, 0]
plt.plot(xx, np.zeros_like(xx), '.-', markersize=20)

msh_ref = mesh.refine(msh)[0]
xx = msh_ref.geometry.x[:, 0]
plt.plot(xx, np.zeros_like(xx), 'o-')
plt.show()


