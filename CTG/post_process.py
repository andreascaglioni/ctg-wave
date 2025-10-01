from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem, io, mesh
import sys
sys.path.insert(0, "./")
from CTG.FE_spaces import TimeFE
from CTG.utils import cart_prod_coords
from mpl_toolkits.mplot3d import Axes3D


def float_f(x):
    """
    Format a float variable in scientific notation.

    Args:
        x (float): Input float.

    Returns:
        str: Formatted float as a string.
    """
    return f"{x:.4e}"


def compute_rate(xx, yy):
    """
    Compute the logarithmic rate of change between consecutive elements of two arrays.

    Args:
        xx (numpy.ndarray): 1D array of x-coordinates.
        yy (numpy.ndarray): 1D array of y-coordinates.

    Returns:
        numpy.ndarray: Logarithmic rates of change.
    """

    return np.log(yy[1:] / yy[:-1]) / np.log(xx[1:] / xx[:-1])

def export_xdmf(msh, f, tt=np.array([]), filename="plot.xdmf"):
    """
    Exports a mesh and associated functions to an XDMF file.

    Args:
        msh (Mesh): The mesh to export.
        f (Function or list of Function): The function(s) to export.
        tt (numpy.ndarray, optional): Time steps for the functions. Defaults to an empty array.
        filename (str, optional): Name of the output XDMF file. Defaults to "plot.xdmf".

    Raises:
        TypeError: If `f` is not a Function or a list of Functions.
    """
    xdmf = io.XDMFFile(msh.comm, filename, "w")
    xdmf.write_mesh(msh)
    if type(f) is list and type(f[0]) is fem.Function:
        if tt.size == 0:
            Warning("export_xdmf: Missing time tt. Using 1,2,...")
            tt = np.linspace(0, len(f) - 1, len(f))
        # export in sequence
        for i in range(len(f)):
            f[i].name = "f"
            xdmf.write_function(f[i], tt[i])
    elif type(f) is fem.Function:
        f.name = "f"
        xdmf.write_function(f)
    else:
        raise TypeError("f has unknown type for export")
    xdmf.close()


def plot_basis_functions(msh_x, V_x):
    dim_V_x = V_x.tabulate_dof_coordinates().shape[0]
    bf = fem.Function(V_x)  # piecewise linear!
    for i in range(dim_V_x):
        dofs_bf = np.zeros((dim_V_x))
        dofs_bf[i] = 1.0
        bf.x.array[:] = dofs_bf
        export_xdmf(msh_x, bf, filename=f"bf{i}.xdmf")


def plot_uv_tt(time_slabs, space_fe, sol_slabs, exact_sol_u=None, exact_sol_v=None):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(sol_slabs[0].size)
    n_dofs_scalar = int(sol_slabs[0].size / 2)

    # Compute bounds y axis
    uu = np.array([X[0:n_dofs_scalar] for X in sol_slabs])
    umin = np.amin(uu)
    umax = np.amax(uu)
    vv = np.array([X[n_dofs_scalar:] for X in sol_slabs])
    vmin = np.amin(vv)
    vmax = np.amax(vv)

    plt.figure(figsize=(10, 4))
    for i, slab in enumerate(time_slabs):
        tx = cart_prod_coords(np.array([slab[0]]), space_fe.dofs)
        X = sol_slabs[i]
        plt.clf()

        # Plot u on the left subplot
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(space_fe.dofs, X[0:n_x], ".", label=f"u at t={round(slab[0], 4)}")
        if exact_sol_u is not None:
            ax1.plot(space_fe.dofs, exact_sol_u(tx), "-", label="u exact")
        ax1.set_title(f"u at t={round(slab[0], 4)}")
        ax1.legend()
        ax1.set_ylim((umin, umax))

        # Plot v on the right subplot
        ax2 = plt.subplot(1, 2, 2)
        vv = X[n_dofs_scalar:n_dofs_scalar+n_x]
        ax2.plot(space_fe.dofs, vv, ".", label=f"v at t={round(slab[0], 4)}")
        if exact_sol_v is not None:
            ax2.plot(space_fe.dofs, exact_sol_v(tx), "-", label="v exact")
        ax2.set_title(f"v at t={round(slab[0], 4)}")
        ax2.legend()
        ax2.set_ylim((vmin, vmax))
        plt.tight_layout()
        
        plt.pause(0.1)


def plot_error_tt(time_slabs, err_slabs, norm_u_slabs):
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker='o', label="error")
    plt.plot(times, rel_errs, marker='o', label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()


def compute_energy_tt(space_fe, sol_slabs):
    """
    Compute total, potential, and kinetic energy over time for a given solution.
    Args:
        space_fe: Finite element space object containing mass and stiffness matrices.
        sol_slabs: List of solution vectors at each time step.
        tt: Array of time points. Each point corresponds to  end of a time slab.
    Returns:
        Tuple of arrays: (total energy, potential energy, kinetic energy) at each time step.
    """

    M =space_fe.matrix["mass"]  # mass
    A = space_fe.matrix["laplace"]  # stiffness
    n_x = space_fe.n_dofs
    n_scalar = int(sol_slabs[0].size/2)
    eenergy = np.zeros(len(sol_slabs))
    ppot = np.zeros(len(sol_slabs))  # potential energy
    kkin = np.zeros(len(sol_slabs))  # kinetic energy

    for i, X in enumerate(sol_slabs):
        u = X[0:n_x]
        v = X[n_scalar:n_scalar+n_x]
        ppot[i] = u @ A @ u
        kkin[i] = v @ M @ v
        eenergy[i] = ppot[i] + kkin[i]
        
    return eenergy, ppot, kkin




def plot_on_slab(dofs_x, dofs_t, X):
    A, B = np.meshgrid(dofs_t, dofs_x, indexing='ij')
    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = X.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(A_flat, B_flat, C_flat, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Time (a)')
    ax.set_ylabel('Space (b)')
    ax.set_zlabel('Value (c)')
    plt.tight_layout()
    plt.show()