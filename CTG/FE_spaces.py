from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
import scipy.sparse
from ufl import TestFunction, TrialFunction, dx, grad, inner


class SpaceFE:
    def __init__(self, V, boundary_D=None):
        # Sanity check input
        assert V.value_size == 1  # always give 1d condomain FE space

        self.mesh = V.mesh
        self.V = V
        self.form = {}
        self.matrix = {}
        
        dofs_raw = self.V.tabulate_dof_coordinates()
        # NB dofs are always 3d! -> Truncate
        gdim = self.mesh.geometry.dim
        self.dofs = dofs_raw[:, 0 : gdim].reshape((-1, gdim))

        self.n_dofs = self.dofs.shape[0]

        self.assemble_matrices()
        
        # initialize self.boundary_dof_vector (if boundary given)
        if boundary_D is not None:
            self.compute_bd_dofs(boundary_D)  

    def assemble_matrices(self):
        u = TrialFunction(self.V)
        phi = TestFunction(self.V)

        self.form["laplace"] = fem.form(inner(grad(u), grad(phi)) * dx)
        self.form["mass"] = fem.form(inner(u, phi) * dx)

        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs, self.n_dofs),
            )

    def compute_bd_dofs(self, boundary_D):
        u_D = fem.Function(self.V)
        u_D.interpolate(lambda x : 0. * x[0])  # dummy boundary data, need only dofs  # noqa: F811
        dofs_boundary = fem.locate_dofs_geometrical(self.V, boundary_D)
        bc = fem.dirichletbc(value=u_D, dofs=dofs_boundary)

        # Compute *indicator function* of boundary
        self.boundary_dof_vector = np.zeros((self.n_dofs * self.V.value_size,))
        for i in bc.dof_indices()[0]:  # bc.dof_indices() is 2-tuple
            self.boundary_dof_vector[i] = 1.0


class TimeFE:
    def __init__(self, mesh, V_trial, V_test):
        assert V_trial.value_size == 1 and V_test.value_size == 1

        self.form = {}
        self.matrix = {}
        self.mesh = mesh
        
        # Trial space
        gdim = mesh.geometry.dim
        self.V_trial = V_trial
        self.dofs_trial = self.V_trial.tabulate_dof_coordinates()[:, 0 : gdim].reshape((-1, gdim))
        self.n_dofs_trial = self.dofs_trial.shape[0]

        # Test space
        self.V_test = V_test
        self.dofs_test = self.V_test.tabulate_dof_coordinates()[:, 0 : mesh.geometry.dim].reshape((-1, gdim))
        self.n_dofs_test = self.dofs_test.shape[0]
        
        # Compute *indicator functions* of IC and FC (Final condition)
        self.dof_IC_vector = np.zeros(self.n_dofs_trial)
        self.dof_IC_vector[np.argmin(self.dofs_trial)] = 1.

        self.dof_FC_vector = np.zeros(self.n_dofs_trial)
        self.dof_FC_vector[np.argmax(self.dofs_trial)] = 1.

        self.assemble_matrices()



    # def get_IC_dofs(self, boundary_IC):
    #     u_IC = fem.Function(self.V_trial)
    #     u_IC.interpolate(lambda tt  : 0. * tt[0])
    #     dofs_IC = fem.locate_dofs_geometrical(self.V_trial, boundary_IC)
    #     ic = fem.dirichletbc(value=u_IC, dofs=dofs_IC)
    #     indic_t0_trial = np.zeros((self.n_dofs_trial,))
    #     for i in ic.dof_indices()[0]:
    #         indic_t0_trial[i] = 1.0
        
    #     u_IC = fem.Function(self.V_test)
    #     u_IC.interpolate(lambda tt  : 0. * tt[0])
    #     dofs_IC = fem.locate_dofs_geometrical(self.V_test, boundary_IC)
    #     ic = fem.dirichletbc(value=u_IC, dofs=dofs_IC)
    #     indic_t0_test = np.zeros((self.n_dofs_test,))
    #     for i in ic.dof_indices()[0]:
    #         indic_t0_test[i] = 1.0

    #     return indic_t0_trial, indic_t0_test

    def print_dofs(self):
        print("\nTime DoFs TRIAL:")
        for dof, dof_t in zip(self.V_trial.dofmap().dofs(), self.dofs_trial):
            print(dof, ":", dof_t)
        print("\nTime DoFs TEST:")
        for dof, dof_t in zip(self.V_test.dofmap().dofs(), self.dofs_test):
            print(dof, ":", dof_t)

    def assemble_matrices(self):
        u = TrialFunction(self.V_trial)
        phi = TestFunction(self.V_test)

        self.form["mass"] = fem.form(inner(u, phi) * dx)
        self.form["derivative"] = fem.form((grad(u)[0] * phi) * dx)
        # self.form["derivative"] = fem.form(inner(grad(u)[0], phi) * dx)
                
        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # TODO why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs_test, self.n_dofs_trial),
            )
