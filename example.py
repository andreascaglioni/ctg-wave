"""CTG approximation wave equation.
Some material is taken from
https://github.com/mathmerizing/SpaceTimeFEM_2023-2024/blob/main/Exercise3/Exercise_3_Linear_PDE.ipynb
"""

from os import join
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, io, mesh
from ufl import dx, grad, inner, TrialFunction, TestFunction


def cart_prod_coords(t_coords, x_coords):
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


class SpaceFE:
    form = {}
    matrix = {}

    def __init__(self, msh, V):
        self.msh = msh
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates().reshape(
            (-1, msh.geometry().dim())
        )
        self.n_dofs = self.dofs.shape[0]
        # For debugging:
        # self.print_dofs()
        self.assemble_matrices()

    def print_dofs(self):
        print("\nSpace DoFs:")
        for dof, dof_x in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_x)

    def assemble_matrices(self):
        u = TrialFunction(self.V)
        phi = TestFunction(self.V)
        self.form["laplace"] = inner(grad(u), grad(phi)) * dx
        self.form["mass"] = u * phi * dx
        for name, _form in self.form.items():
            self.matrix[name] = scipy.sparse.csr_matrix(
                (fem.petsc.assemble(_form)).mat().getValuesCSR()[::-1],
                shape=(self.n_dofs, self.n_dofs),
            )


class TimeFE:
    form = {}
    matrix = {}

    def __init__(self, mesh, V):
        self.mesh = mesh
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates().reshape((-1, 1))
        self.n_dofs = self.dofs.shape[0]
        # For debugging:
        # self.print_dofs()
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
        self.form["derivative"] = (
            grad(u)[0] * phi * dx
            # NB I do CTG therefore dont need these 2
            # + (u("-") - u("+")) * phi("-") * dS  # jump interior facets
            # + u("+") * phi("+") * d0  # jump initial time
        )
        self.form["mass"] = u * phi * dx

        for name, _form in self.form.items():
            self.matrix[name] = scipy.sparse.csr_matrix(
                (fem.petsc.assemble(_form)).mat().getValuesCSR()[::-1],
                shape=(self.n_dofs, self.n_dofs),
            )


if __name__ == "__main__":
    comm = MPI.COMM_SELF
    L2_error = 0.0
    total_temporal_dofs = 0

    # Create space FE object
    msh_x = mesh.create_unit_interval(comm, 10, 0, 1)
    V_x = fem.FunctionSpace(msh_x, ("CG", 1))
    Space = SpaceFE(msh_x, V_x)

    # Time marching
    start_time = 0.0
    end_time = 0.5
    slab_size = 0.01  # 0.05 # 0.25
    slabs = [(start_time, start_time + slab_size)]
    while slabs[-1][1] < end_time - 1e-8:
        slabs.append((slabs[-1][1], slabs[-1][1] + slab_size))
    print(f"\nSlabs = {slabs}")

    # define exact solution and rhs function
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

    u0 = exact_sol(cart_prod_coords(np.array([[start_time]]), Space.dofs))

    # compute spatial boundary DoF indices
    bc = fem.DirichletBC(Space.V, fem.Constant(1.0), lambda _, on_boundary: on_boundary)
    Space.boundary_dof_vector = np.zeros((Space.n_dofs,))
    for i in bc.get_boundary_values().keys():
        Space.boundary_dof_vector[i] = 1.0

    # Compute
    for i, slab in enumerate(slabs):
        print(
            f"Solving on slab_{i} = Ω x ({round(slab[0], 5)}, {round(slab[1], 5)}) ..."
        )

        # t FE object
        r = 1  # polynomial degree in time
        n_time = 4  # number of temporal elements
        t_msh = mesh.create_interval(n_time, slab[0], slab[1])
        # Start time: slab[0], End time: slab[1] = slab[0]+slab_size
        V_t = fem.FunctionSpace(t_msh, "DG", r)

        # Temporal FE object:
        Time = TimeFE(t_msh, V_t)
        total_temporal_dofs += Time.n_dofs

        # Assemble space-time matrices
        # NOTE: linear PDEs -> kronecker product temporal & spatial matrices
        system_matrix = scipy.sparse.kron(
            Time.matrix["derivative"], Space.matrix["mass"]
        ) + scipy.sparse.kron(Time.matrix["mass"], Space.matrix["laplace"])

        mass_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["mass"])

        # Assemble right hand side vector
        # 1. evaluate rhs fun at space-time Dofs and
        # 2. define the RHS as: space-time mass * proj RHS fun
        # TODO better to use projection? I'll use higher order FEM!
        space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
        rhs = mass_matrix.dot(exact_rhs(space_time_coords))
        # add weak imposition of the initial condition to the right hand side
        rhs[: Space.n_dofs] += Space.matrix["mass"].dot(u0)

        # Apply boundary conditions
        # set the analytical solution as Dirichlet boundary conditions on the entire spatial boundary
        dofs_at_boundary = np.kron(
            np.ones((Time.dofs.shape[0], 1)), Space.boundary_dof_vector.reshape(-1, 1)
        ).flatten()
        slab_exact_sol = exact_sol(space_time_coords)

        # apply space-time BC to system matrix
        system_matrix = system_matrix.multiply((1.0 - dofs_at_boundary).reshape(-1, 1))
        +scipy.sparse.diags(dofs_at_boundary)

        # apply space-time BC to right hand side
        rhs = rhs * (1.0 - dofs_at_boundary) + slab_exact_sol * dofs_at_boundary

        #Solve linear system
        # solve the linear system with a sparse direct solver
        slab_solution = scipy.sparse.linalg.spsolve(system_matrix, rhs)

        # Compute error to analytical solution 
        # error_vector: proj exact sol - the FEM sol
        slab_error = slab_exact_sol - slab_solution
        # L^2(I, L^2(Ω)) error on the slab
        L2_error += mass_matrix.dot(slab_error).dot(slab_error)

        # get initial condition on next slab = final condition from this slab
        last_time_dof = Time.dofs.argmax()
        u0 = slab_solution[
            last_time_dof * Space.n_dofs : (last_time_dof + 1) * Space.n_dofs
        ]
        print("Done.\n")
