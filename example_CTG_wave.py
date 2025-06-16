"""CTG approximation wave equation. Some material is taken from

https://github.com/mathmerizing/SpaceTimeFEM_2023-2024/blob/main/Exercise3/Exercise_3_Linear_PDE.ipynb

"""

from math import sqrt
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, io, mesh, default_real_type
from ufl import dx, grad, inner, TrialFunction, TestFunction
from basix import ufl
from dolfinx.fem.petsc import assemble_matrix


def cart_prod_coords(t_coords, x_coords):
    if len(x_coords.shape) == 1:
        x_coords = np.expand_dims(x_coords, 1)

    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


class SpaceFE:
    form = {}
    matrix = {}

    def __init__(self, msh, V):
        self.msh = msh
        self.V = V
        # self.dofs = self.V.tabulate_dof_coordinates().reshape((-1, msh.geometry.dim))
        self.dofs = self.V.tabulate_dof_coordinates()
        self.dofs = self.dofs[:, 0].reshape((-1, 1))

        # NB the return is always a list of length 3 tuples: [(x1, x2, x3), ...]. If geometric dimension <3, the x2=x3=1!!!
        self.n_dofs = self.dofs.shape[0]
        # self.print_dofs()  # debugging
        self.assemble_matrices()

    def print_dofs(self):
        print("\nSpace DoFs:")
        for dof, dof_x in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_x)

    def assemble_matrices(self):
        u = TrialFunction(self.V)
        phi = TestFunction(self.V)

        self.form["laplace"] = fem.form(inner(grad(u), grad(phi)) * dx)
        self.form["mass"] = fem.form(inner(u, phi) * dx)

        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs, self.n_dofs),
            )


class TimeFE:
    form = {}
    matrix = {}

    def __init__(self, mesh, V):
        self.mesh = mesh
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates()
        self.dofs = self.dofs[:, 0].reshape((-1, 1))
        # self.dofs = self.V.tabulate_dof_coordinates().reshape((-1, 1))  # WRONG: coords are always 3d. Flatten triples the # dofs

        self.n_dofs = self.dofs.shape[0]
        # self.print_dofs()  # debugging
        self.assemble_matrices()

    def print_dofs(self):
        print("\nTime DoFs:")
        for dof, dof_t in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_t)

    def assemble_matrices(self):
        # initial_time = CompiledSubDomain("near(x[0], t0)", t0=self.dofs[0, 0])
        # interior_facets = CompiledSubDomain("!on_boundary")
        # boundary_marker = MeshFunction("size_t", self.mesh, 0)
        # boundary_marker.set_all(0)
        # initial_time.mark(boundary_marker, 1)
        # interior_facets.mark(boundary_marker, 2)

        # Measure for the initial time
        # d0 = Measure(
        #     "ds", domain=self.mesh, subdomain_data=boundary_marker, subdomain_id=1
        # )
        # dS = Measure(
        #     "dS", domain=self.mesh, subdomain_data=boundary_marker, subdomain_id=2
        # )

        u = TrialFunction(self.V)
        phi = TestFunction(self.V)

        # NOTE: FEniCS has weird definitions for '+' and '-' (https://fenicsproject.discourse.group/t/integrating-over-an-interior-surface/247/3)

        self.form["derivative"] = fem.form(grad(u)[0] * phi * dx)
        # NB I do CTG therefore dont need fcollowing 2 lines
        # + (u("-") - u("+")) * phi("-") * dS  # jump interior facets
        # + u("+") * phi("+") * d0  # jump initial time
        self.form["mass"] = fem.form(u * phi * dx)

        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # TODO why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs, self.n_dofs),
            )
            # self.matrix[name] = scipy.sparse.csr_matrix(
            #     (fem.petsc.assemble(_form)).mat().getValuesCSR()[::-1],
            #     shape=(self.n_dofs, self.n_dofs),
            # )


def compute_time_slabs(start_time, end_time, slab_size):
    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs


if __name__ == "__main__":
    # ------------------------------------------------------------------------ #
    #                                   DATA                                   #
    # ------------------------------------------------------------------------ #
    comm = MPI.COMM_SELF
    L2_error = 0.0
    total_temporal_dofs = 0

    # ----------------------------- Numerics data ---------------------------- #
    # Space FEM
    msh_x = mesh.create_unit_interval(comm, 10)
    order_x = 1
    Elements_x = ufl.element("Lagrange", msh_x.basix_cell(), order_x)
    V_x = fem.functionspace(msh_x, Elements_x)

    # Time marching & FEM
    slab_size = 0.1  # 0.05 # 0.25
    order_t = 1  # polynomial degree in time

    n_time = 10  # number of temporal elements per time-slab

    # ----------------------------- Physical data ---------------------------- #
    start_time = 0.0
    end_time = 0.5

    def exact_sol(X):
        _x = X[:, 1]
        _t = X[:, 0]
        return np.sin(np.pi * _x) * (1.0 + _t) * np.exp(-0.5 * _t)

    def exact_rhs(X):
        _x = X[:, 1]
        _t = X[:, 0]
        return (
            np.sin(np.pi * _x)
            * np.exp(-0.5 * _t)
            * (0.5 + np.pi**2 + (np.pi**2 - 0.5) * _t)
        )

    # ------------------------------------------------------------------------ #
    #                                  COMPUTE                                 #
    # ------------------------------------------------------------------------ #

    # Time slabs
    time_slabs = compute_time_slabs(start_time, end_time, slab_size)

    # FE object for SPACE discretization
    Space = SpaceFE(msh_x, V_x)
    u0 = exact_sol(cart_prod_coords(np.array([[start_time]]), Space.dofs))

    # Dirichlet BCs
    u_D = fem.Function(V_x)
    u_D.interpolate(exact_sol)  # u_D.interpolate(lambda x: 0.0 * x[0])
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    dofs_boundary = fem.locate_dofs_geometrical(V_x, boundary_D)
    bc = fem.dirichletbc(value=u_D, dofs=dofs_boundary)

    # Space.boundary_dof_vector is "indicator function" of boundary: 1 on booundary, 0 otherwise
    Space.boundary_dof_vector = np.zeros((Space.n_dofs,))
    for i in bc.dof_indices()[0]:  # NB bc.dof_indices() is 2-tuple
        Space.boundary_dof_vector[i] = 1.0

    # Time marching
    for i, slab in enumerate(time_slabs):
        print(
            f"Solving on slab_{i} = D x ({round(slab[0], 5)}, {round(slab[1], 5)}) ...",
            flush=True,
        )

        # Compute FE object for TIME discretization
        msh_t = mesh.create_interval(comm, n_time, [slab[0], slab[1]])
        Elements_t = ufl.element("Lagrange", msh_x.basix_cell(), order_t)
        V_t = fem.functionspace(msh_t, Elements_t)
        Time = TimeFE(msh_t, V_t)
        total_temporal_dofs += Time.n_dofs

        # Assemble space-time matrices (linear PDEs -> Kronecker product t & x matrices)
        system_matrix = scipy.sparse.kron(
            Time.matrix["derivative"], Space.matrix["mass"]
        )
        system_matrix += scipy.sparse.kron(Time.matrix["mass"], Space.matrix["laplace"])
        
        mass_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["mass"])

        # Assemble right hand side vector
        # 1. evaluate rhs fun at space-time Dofs
        # 2. define the RHS as: space-time mass * proj RHS fun TODO better to use projection? I'll use higher order FEM!
        # 3. add weak imposition of the initial condition to the right hand side
        space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
        rhs = mass_matrix.dot(exact_rhs(space_time_coords))
        rhs[: Space.n_dofs] += Space.matrix["mass"].dot(u0)

        # Apply boundary conditions (use exact sol)
        # Idea: modify A and RHS so that, if the i-th dof belongs to the boundary,
        # then the i-th equation enforces the BC instead of the equation wrt the
        # i-th test function. This means that
        # - A i s modified so that the i-th row decomes a delta_{i,j}
        # - RHS is modified so that the i-th entry is the BC on i-th dof

        # 1. recover data
        dofs_at_boundary = np.kron(
            np.ones((Time.dofs.shape[0], 1)), Space.boundary_dof_vector.reshape(-1, 1)
        ).flatten()
        slab_exact_sol = exact_sol(space_time_coords)

        # 2. apply space-time BC to system matrix
        system_matrix = system_matrix.multiply(
            (1.0 - dofs_at_boundary).reshape(-1, 1)
        )  # put to 0 entries corresponding to boundary
        system_matrix += scipy.sparse.diags(dofs_at_boundary)

        # 3. apply space-time BC to RHS vector
        rhs = rhs * (1.0 - dofs_at_boundary)
        rhs += slab_exact_sol * dofs_at_boundary

        # Solve linear system (sparse direct solver)
        slab_solution = scipy.sparse.linalg.spsolve(system_matrix, rhs)
        # Check residual
        residual_slab = system_matrix.dot(slab_solution) - rhs
        rel_res_slab = np.linalg.norm(residual_slab) / np.linalg.norm(slab_solution)
        print(f"Relative residual solver slab {i}: {rel_res_slab}")

        # Error curr slab
        slab_error = slab_exact_sol - slab_solution
        L2_error += mass_matrix.dot(slab_error).dot(slab_error)

        # get initial condition on next slab = final condition from this slab
        last_time_dof = Time.dofs.argmax()
        u0 = slab_solution[
            last_time_dof * Space.n_dofs : (last_time_dof + 1) * Space.n_dofs
        ]
        print("Done.\n")


        # Plot exact and numerical solution at t=slab[0]
        fig, ax = plt.subplots()
        first_time_idx = 0
        x_vals = Space.dofs.flatten()
        # Numerical solution at first time
        numerical = slab_solution[
            first_time_idx * Space.n_dofs : (first_time_idx + 1) * Space.n_dofs
        ]
        # Exact solution at first time
        space_time_first = cart_prod_coords(
            Time.dofs[first_time_idx:first_time_idx+1], Space.dofs
        )
        exact = exact_sol(space_time_first)
        ax.plot(x_vals, numerical, label="Numerical", marker='o')
        ax.plot(x_vals, exact, label="Exact", marker='x')
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"Slab {i}, t={Time.dofs[first_time_idx,0]:.3f}")
        ax.legend()
        plt.show()

    print("L2 error", sqrt(L2_error))
