# from https://fenicsproject.discourse.group/t/convergence-problem-of-linear-wave-equation/14432
# Solve on D = [0,1] and for T = 2 * np.pi + 0.5 wave equation
#
# \partial_{tt} u - \Delta u = 0
# u = v = 0 in \partial D
# u(0) = sin(pi*x), v(0) = 0
#
# with exact sol
#
# u = sin(pi*x) * cos(pi*t)
# v = -pi * sin(pi*x) * sin(pi*t)
#


from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx import mesh, fem, plot
import ufl
import basix
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
import pyvista


# error
def error_L2(uh, u_ex, degree_raise=3):
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().element_family.name
    mesh = uh.function_space.mesh
    W = fem.functionspace(
        mesh, (family, degree + degree_raise)
    )  # high-dim space for error function

    # interpolate
    u_W = fem.Function(W)
    u_W.interpolate(uh)
    u_ex_W = fem.Function(W)
    u_ex_W.interpolate(u_ex)

    # Compute error
    e_W = fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


# exact solution
class V_exact:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        return np.sin(np.pi * x[0]) * np.cos(np.pi * self.t)


def plot_sol(transparent, figsize, Space_u, u):
    cells, types, x = plot.vtk_mesh(Space_u)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array
    grid.set_active_scalars("u")
    pvplot = pyvista.Plotter()
    pvplot.add_text(
        "Scalar contour field", font_size=14, color="black", position="upper_edge"
    )
    pvplot.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    pvplot.view_xy()
    if pyvista.OFF_SCREEN:
        pvplot.screenshot(
            "2D_function_warp.png",
            transparent_background=transparent,
            # window_size=[figsize, figsize],
        )
    else:
        pvplot.show()


if __name__ == "__main__":
    # Nx = 40
    errors = np.array([])
    hlist = np.array([])

    scalings = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1042, 2048]
    for i in range(len(scalings) - 3):
        # parameters
        Nx = 1 * scalings[i] * 1
        Ny = 1 * scalings[i] * 1
        Nt = 1 * scalings[i] * 1
        # Nt = 1024 # UNCOMMENT THIS
        T_end = 2 * np.pi + 0.5
        h = 1 / Nx
        dt = T_end / Nt
        hlist = np.append(hlist, h)
        t = 0
        pdeg = 1  # CHANGE TO 2

        # mesh, spaces, functions
        msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Nx)
        x = ufl.SpatialCoordinate(msh)
        n = ufl.FacetNormal(msh)
        P1 = basix.ufl.element("Lagrange", "triangle", pdeg)
        XV = fem.functionspace(msh, P1)
        Xp = fem.functionspace(msh, P1)
        V = ufl.TrialFunction(XV)
        p = ufl.TrialFunction(Xp)
        W = ufl.TestFunction(XV)
        q = ufl.TestFunction(Xp)
        V_old = fem.Function(XV)
        V_new = fem.Function(XV)
        p_old = fem.Function(Xp)
        p_new = fem.Function(Xp)
        b = fem.Constant(msh, (1.0, 0.0))

        # init cond
        def initcond_V(x):
            return np.sin(np.pi * x[0])

        def initcond_p(x):
            return 0 + 0.0 * x[0]

        V_old.interpolate(initcond_V)
        p_old.interpolate(initcond_p)
        V_new.interpolate(initcond_V)
        p_new.interpolate(initcond_p)

        # BC
        facetsV = mesh.locate_entities_boundary(
            msh,
            dim=1,
            marker=lambda x: np.logical_or.reduce(
                (np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
            ),
        )
        dofsV = fem.locate_dofs_topological(XV, entity_dim=1, entities=facetsV)
        BCs = [fem.dirichletbc(0.0, dofsV, XV)]

        # weak form
        M = ufl.inner(V, W) * ufl.dx
        E = ufl.inner(ufl.grad(p), b * W) * ufl.dx
        F = ufl.inner(ufl.grad(V), b * q) * ufl.dx
        N = ufl.inner(p, q) * ufl.dx
        a = fem.form([[M, -dt / 2 * E], [-dt / 2 * F, N]])
        A = assemble_matrix_block(a, bcs=BCs)
        A.assemble()
        Vp_vec = A.createVecLeft()

        # solver
        solver = PETSc.KSP().create(msh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        #  Forms
        l = fem.form(
            [
                ufl.inner(V_old, W) * ufl.dx
                + dt / 2 * ufl.inner(ufl.grad(p_old), b * W) * ufl.dx,
                ufl.inner(p_old, q) * ufl.dx
                + dt / 2 * ufl.inner(ufl.grad(V_old), b * q) * ufl.dx,
            ]
        )
        L = assemble_vector_block(l, a, bcs=BCs)
        
        for i in range(Nt):
            t += dt

            # Update the right hand side
            

            # solve
            solver.solve(L, Vp_vec)

            # extract solution
            offset = XV.dofmap.index_map.size_local * XV.dofmap.index_map_bs
            V_new.x.array[:offset] = Vp_vec.array_r[:offset]
            p_new.x.array[: (len(Vp_vec.array_r) - offset)] = Vp_vec.array_r[offset:]

            # Update solution at previous time step
            V_old.x.array[:] = V_new.x.array
            p_old.x.array[:] = p_new.x.array

            plot_sol(True, figsize=None, Space_u=XV, u=V_new)

        # error in high-dim polynomial space
        V_ex_expr = V_exact()
        V_ex_expr.t = T_end
        error = error_L2(V_new, V_ex_expr.eval, degree_raise=5)
        errors = np.append(errors, error)
        print(error)

    # print errors and convergence rate
    # print(errors)
    rates = np.log(errors[1:] / errors[:-1]) / np.log(hlist[1:] / hlist[:-1])
    print(rates)
