from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
import scipy.sparse
from ufl import TestFunction, TrialFunction, dx, grad, inner


class SpaceFE:
    def __init__(self, mesh, V, boundary_data=None, boundary_D=None):

        assert (boundary_data is None and boundary_D is None) or (
            boundary_data is not None and boundary_D is not None
        )
        self.form = {}
        self.matrix = {}
        self.mesh = mesh
        self.V = V

        self.dofs = self.V.tabulate_dof_coordinates()
        # NB dofs are always 3d! -> Truncate
        # TODO handle different dimensions space domain
        self.dofs = self.dofs[:, 0 : mesh.geometry.dim].reshape((-1, mesh.geometry.dim))  

        self.n_dofs = self.dofs.shape[0]
        # self.print_dofs()  # debugging
        self.assemble_matrices()

        if boundary_data is not None:
            self.set_boundary_conditions(boundary_data, boundary_D)

    def set_boundary_conditions(self, boundary_data, boundary_D):
        u_D = fem.Function(self.V)

        # express boundary data (function of TX) as function of X only
        def boundary_data_x(xx):
            tt = np.zeros((xx.shape[1], 1))
            X = np.hstack((tt, xx.T))
            return boundary_data(X)

        u_D.interpolate(boundary_data_x)
        dofs_boundary = fem.locate_dofs_geometrical(self.V, boundary_D)
        bc = fem.dirichletbc(value=u_D, dofs=dofs_boundary)
        # TODO look at BC

        # Compute "indicator function" of boundary: 1 on boundary, 0 otherwise

        self.boundary_dof_vector = np.zeros((self.n_dofs*self.V.value_size,))
        for i in bc.dof_indices()[0]:  # bc.dof_indices() is 2-tuple
            self.boundary_dof_vector[i] = 1.0

    def print_dofs(self):
        print("\nSpace DoFs:")
        for dof, dof_x in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_x)

    def assemble_matrices(self):  # NB assemble matrices scalar components
        if self.V.value_size > 1:
            Vc = fem.functionspace(self.mesh, ("Lagrange", 1) )  # TODO handle different too!

            geo_dim = self.mesh.geometry.dim
            dc = self.dofs[:, 0 : geo_dim].reshape((-1, geo_dim))  
            nd = dc.shape[0]
        else:
            Vc = self.V
            nd = self.n_dofs

        u = TrialFunction(Vc)
        phi = TestFunction(Vc)
        
        self.form["laplace"] = fem.form(inner(grad(u), grad(phi)) * dx)
        self.form["mass"] = fem.form(inner(u, phi) * dx)
        
        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(nd, nd),
            )


class TimeFE:
    def __init__(self, mesh, V_trial, V_test):
        self.form = {}
        self.matrix = {}
        self.mesh = mesh
        
        self.V_trial = V_trial
        self.dofs_trial = self.V_trial.tabulate_dof_coordinates()
        self.dofs_trial = self.dofs_trial[:, 0 : mesh.geometry.dim].reshape((-1, mesh.geometry.dim))
        self.n_dofs_trial = self.dofs_trial.shape[0]
        
        self.V_test = V_test
        self.dofs_test = self.V_test.tabulate_dof_coordinates()
        self.dofs_test = self.dofs_test[:, 0 : mesh.geometry.dim].reshape((-1, mesh.geometry.dim))
        self.n_dofs_test = self.dofs_test.shape[0]
        
        # self.print_dofs()  # debugging
        self.assemble_matrices()
 

    def get_IC_dofs(self, initial_data, boundary_IC):
        """Get the indices of dofs correspoding to IC, for both test and trial spaces.

        Args:
            initial_data (Callable): Functiopn of tx encoding IC (constant in t)
            boundary_IC (_type_): Functiontat given array (gdim, n) return Bool array (n, ) with True for IC boundary

        Returns:
            array: Length 1 array con taining the unique index of the dof correspodning to IC in V_trial
        """
        u_IC = fem.Function(self.V_trial)

        # express boundary data (function of TX) as function of X only
        def initial_data_t(tt_in):
            # coming from self.V_trial, tt is of shape (3, n_tt) with actual time only in ***1st ROW***
            xx = np.zeros(tt_in.shape)
            tt = tt_in[0, :].reshape((-1, 1))  # remove last tow ROWS for fictitiouss dims 1,2
            X = np.hstack((tt, xx.T))
            return initial_data(X)

        # def boundary_data_x(xx):
        #     tt = np.zeros((xx.shape[1], 1))
        #     X = np.hstack((tt, xx.T))
        #     return boundary_data(X)


        u_IC.interpolate(initial_data_t)
        dofs_IC = fem.locate_dofs_geometrical(self.V_trial, boundary_IC)
        ic = fem.dirichletbc(value=u_IC, dofs=dofs_IC)
    
        # Compute "indicator function" of boundary: 1 on boundary, 0 otherwise
        ic_dof_vector = np.zeros((self.n_dofs*self.V.value_size,))
        for i in ic.dof_indices()[0]:
            self.ic_dof_vector[i] = 1.0
        return ic_dof_vector


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

        self.form["derivative"] = fem.form(grad(u)[0] * phi * dx)
        self.form["mass"] = fem.form(u * phi * dx)
        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]  # TODO why -1?
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs_test, self.n_dofs_trial),
            )