"""Generate a sequence of structured unit square meshes with the method from dolfinx.mesh.create_unit_square."""

from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
import numpy as np


nn = 4 * 2 ** np.arange(7)  # number of elements along one edge of the square
for n in nn:
    print("n elements per edge:", n)
    filename = "msh_square_dolfinx/mesh_square_" + str(int(n)) + ".xdmf"
    msh = create_unit_square(MPI.COMM_SELF, n, n)
    xdmf = XDMFFile(msh.comm, filename, "w")
    xdmf.write_mesh(msh)
    xdmf.close()
