import numpy as np
from dolfinx import fem, io


def float_f(x):
    """
    Format a float variable in scientific notation.

    Args:
        x (float): Input float.

    Returns:
        str: Formatted float as a string.
    """
    return f"{x:.4e}"

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