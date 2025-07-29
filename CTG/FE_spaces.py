from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
import scipy.sparse
from ufl import TestFunction, TrialFunction, dx, grad, inner


class SpaceFE:
    form = {}
    matrix = {}

    def __init__(self, mesh, V, boundary_data=None, boundary_D=None):
        assert (boundary_data is None and boundary_D is None) or (
            boundary_data is not None and boundary_D is not None
        )
        self.mesh = mesh
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates()
        # NB dofs are always 3d! -> Truncate
        self.dofs = self.dofs[:, 0 : mesh.geometry.dim].reshape((-1, mesh.geometry.dim))
        self.n_dofs = self.dofs.shape[0]
        # self.print_dofs()  # debugging
        self.assemble_matrices()

        if boundary_data is not None:
            self.set_boundary_conditions(boundary_data, boundary_D)

    def set_boundary_conditions(self, boundary_data, boundary_D):
        u_D = fem.Function(self.V)

        # express boundary data as a function of space only
        def boundary_data_x(xx):
            tt = np.zeros((xx.shape[1], 1))
            X = np.hstack((tt, xx.T))
            return boundary_data(X)

        u_D.interpolate(boundary_data_x)
        dofs_boundary = fem.locate_dofs_geometrical(self.V, boundary_D)
        bc = fem.dirichletbc(value=u_D, dofs=dofs_boundary)

        # Compute "indicator function" of boundary: 1 on boundary, 0 otherwise
        self.boundary_dof_vector = np.zeros((self.n_dofs,))
        for i in bc.dof_indices()[0]:  # bc.dof_indices() is 2-tuple
            self.boundary_dof_vector[i] = 1.0

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
        # NB dofs are always 3d! -> Truncate
        self.dofs = self.dofs[:, 0 : mesh.geometry.dim].reshape((-1, mesh.geometry.dim))
        self.n_dofs = self.dofs.shape[0]
        # self.print_dofs()  # debugging
        self.assemble_matrices()

    def print_dofs(self):
        print("\nTime DoFs:")
        for dof, dof_t in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_t)

    def assemble_matrices(self):
        u = TrialFunction(self.V)
        phi = TestFunction(self.V)
        self.form["derivative"] = fem.form(grad(u)[0] * phi * dx)
        self.form["mass"] = fem.form(u * phi * dx)
        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # TODO why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs, self.n_dofs),
            )