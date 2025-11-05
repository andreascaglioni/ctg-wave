"""Finite-element spaces and small helpers used by the CTG solver.

This module provides lightweight wrappers around the Dolfinx function
spaces and the small utilities required to assemble space, time and
space-time operators used by the CTG implementation.

Only a subset of behaviour is implemented here (Lagrangian FE and scalar
fields) — the public classes are :class:`SpaceFE`, :class:`TimeFE` and
:class:`SpaceTimeFE`.
"""

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
import scipy.sparse
import ufl
from typing import Dict

from ctg.utils import cart_prod_coords


class SpaceFE:
    """Finite-element space in the spatial domain.

    Args:
        V: A Dolfinx FunctionSpace for the spatial domain (scalar codomain).
        boundary_D: Optional geometrical marker used to compute Dirichlet
            boundary dofs.
    """

    def __init__(self, V: fem.FunctionSpace, boundary_D=None):
        # Sanity check input
        assert V.value_size == 1, f"SpaceFE: Condomain of V must be 1D, not {V.value_size}"
        self.V = V
        self.form: Dict[str, fem.Form] = {}
        self.matrix: Dict[str, scipy.sparse.csr_matrix] = {}
        dofs_raw = self.V.tabulate_dof_coordinates()
        # NB dofs are always 3d! -> Truncate
        gdim = V.mesh.geometry.dim
        self.dofs = dofs_raw[:, 0:gdim].reshape((-1, gdim))
        self.n_dofs = self.dofs.shape[0]
        self.assemble_matrices()
        # initialize self.boundary_dof_vector (if boundary given)
        if boundary_D is not None:
            self.compute_bd_dofs(boundary_D)

    def assemble_matrices(self):
        """Assemble standard FE matrices (mass and laplace).

        The matrices are stored in :attr:`self.matrix` under keys
        ``"mass"`` and ``"laplace"``.
        """
        u = ufl.TrialFunction(self.V)
        phi = ufl.TestFunction(self.V)
        self.form["laplace"] = fem.form(ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx)
        self.form["mass"] = fem.form(ufl.inner(u, phi) * ufl.dx)
        for name, _form in self.form.items():
            dl_mat_curr = assemble_matrix(_form)
            dl_mat_curr.assemble()
            dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]
            self.matrix[name] = scipy.sparse.csr_matrix(
                dl_mat_curr2,
                shape=(self.n_dofs, self.n_dofs),
            )

    def compute_bd_dofs(self, boundary_D):
        """Compute an indicator vector for Dirichlet boundary dofs.

        The resulting vector ``self.boundary_dof_vector`` contains ones at
        boundary dof indices and zeros elsewhere.
        """
        u_D = fem.Function(self.V)
        u_D.interpolate(
            lambda x: 0.0 * x[0]
        )  # dummy boundary data, need only dofs  # noqa: F811 # TODO fix issue above
        dofs_boundary = fem.locate_dofs_geometrical(self.V, boundary_D)
        bc = fem.dirichletbc(value=u_D, dofs=dofs_boundary)
        # Compute *indicator function* of boundary
        self.boundary_dof_vector = np.zeros((self.n_dofs * self.V.value_size,))
        for i in bc.dof_indices()[0]:  # bc.dof_indices() is 2-tuple
            self.boundary_dof_vector[i] = 1.0


class TimeFE:
    # use assemble_matrices_W(W_t) to assemble W_mass, WW_mass
    def __init__(self, V, verbose=False):
        """Finite-element space in the temporal domain.

        Args:
            V: A Dolfinx FunctionSpace defined on a 1D time mesh.
            verbose: If True, print informative messages during assembly.
        """
        assert V.value_size == 1, f"TimeFE: Codomain FE space ust be 1 not {V.value_size}"
        self.form = {}
        self.matrix = {}
        gdim = V.mesh.geometry.dim
        self.V = V
        self.dofs = self.V.tabulate_dof_coordinates()[:, 0:gdim].reshape((-1, gdim))
        self.n_dofs = self.dofs.shape[0]
        self.verbose = False
        # Compute *indicator functions* of IC and FC (final condition)
        self.dof_IC_vector = np.zeros(self.n_dofs)
        self.dof_IC_vector[np.argmin(self.dofs)] = 1.0
        self.dof_FC_vector = np.zeros(self.n_dofs)
        self.dof_FC_vector[np.argmax(self.dofs)] = 1.0
        self.assemble_matrices_0()

    def print_dofs(self):
        print("\nTime DOFs")
        for dof, dof_t in zip(self.V.dofmap().dofs(), self.dofs):
            print(dof, ":", dof_t)

    def _add_form_matrix(self, name, form):
        """Add 1 form to instance and compute the matrix."""
        self.form[name] = form
        dl_mat_curr = assemble_matrix(form)
        dl_mat_curr.assemble()
        dl_mat_curr2 = dl_mat_curr.getValuesCSR()[::-1]
        self.matrix[name] = scipy.sparse.csr_matrix(dl_mat_curr2, shape=(self.n_dofs, self.n_dofs))

    def assemble_matrices_0(self):
        """Assemble W-independent time matrices (mass, derivative).

        These are stored in :attr:`self.matrix` and used by
            :class:`SpaceTimeFE` to form tensor-product operators.
        """
        u = ufl.TrialFunction(self.V)
        phi = ufl.TestFunction(self.V)
        # Mass
        f = fem.form(ufl.inner(u, ufl.grad(phi)[0]) * ufl.dx)
        self._add_form_matrix("mass", f)
        # Classical mass matrix with same test and trial spaces
        f = fem.form(ufl.inner(u, phi) * ufl.dx)
        self._add_form_matrix("mass_err", f)
        # Derivative
        f = fem.form((ufl.grad(u)[0] * ufl.grad(phi)[0]) * ufl.dx)
        self._add_form_matrix("derivative", f)

    def assemble_matrices_W(self, W_t=None):
        """Assemble time matrices that depend on the coefficient W_t.

        Args:
            W_t: Callable evaluated at time dofs. If ``None`` the method is
                a no-op.
        """
        if W_t is None:
            if self.verbose:
                print("TimeFE Warning: W_t is None. Skiping assembly W_dependent operators.")
            return
        u = ufl.TrialFunction(self.V)
        phi = ufl.TestFunction(self.V)
        W_fun = fem.Function(self.V)
        W_fun.interpolate(W_t)
        # Wu
        f = fem.form(ufl.inner(W_fun * u, ufl.grad(phi)[0]) * ufl.dx)
        self._add_form_matrix("W_mass", f)
        # W**2u
        f = fem.form(ufl.inner(W_fun * W_fun * u, ufl.grad(phi)[0]) * ufl.dx)
        self._add_form_matrix("WW_mass", f)
        return self.matrix["W_mass"], self.matrix["WW_mass"]


class SpaceTimeFE:
    """Space-time FE container and operator assembler.

    This class couples a :class:`SpaceFE` and a :class:`TimeFE` and
    constructs the tensor-product matrices used by the CTG solver. Use
    :meth:`update_time_fe` to move between time slabs.
    """

    def __init__(self, space_fe: SpaceFE, time_fe: "TimeFE | None" = None, verbose: bool = False):

        self.space_fe = space_fe
        self.time_fe = time_fe
        self.verbose = verbose
        self.matrix: Dict[str, scipy.sparse.csr_matrix] = (
            {}
        )  # dictionary space time operators matrices
        if type(time_fe) is TimeFE:
            self.update_time_fe(time_fe)
        else:
            self.dofs = np.array([[]])
            self.n_dofs = 0
            self.dofs_IC = np.array([])
            self.dofs_FC = np.array([])

    def update_time_fe(self, time_fe: TimeFE):
        """Update time_fe and related members. Use it moving to a new time slab."""
        self.time_fe = time_fe
        self.dofs = cart_prod_coords(self.time_fe.dofs, self.space_fe.dofs)
        self.n_dofs = self.dofs.shape[0]
        self.update_IC_FC_dofs()

    def update_IC_FC_dofs(self):
        if self.time_fe is None:
            if self.verbose:
                print("Warning: self.time_fe is None. Cannot update IC/FC dofs.")
            return
        n_dofs_x = self.space_fe.n_dofs
        # IC DOFs
        dofs_ic_t = self.time_fe.dof_IC_vector
        dofs_ic_tx_scalar = np.kron(dofs_ic_t, np.ones(n_dofs_x))
        self.dofs_IC = np.tile(dofs_ic_tx_scalar, 2).astype(bool)
        # FC DOFs
        dofs_fc_t = self.time_fe.dof_FC_vector
        dofs_fc_tx_scalar = np.kron(dofs_fc_t, np.ones(n_dofs_x))
        self.dofs_FC = np.tile(dofs_fc_tx_scalar, 2).astype(bool)

    def interpolate(self, f) -> np.ndarray:
        # TODO works only for Lagrangian FEM. Implement projection
        return f(self.dofs)

    def assemble(self, W_y):
        """Assemble all operators."""
        self.assemble_noW()
        self.assemble_W(W_y)

    def assemble_noW(self):
        """Assemble opeartors independent of parameter y."""
        if self.time_fe is None:
            if self.verbose:
                print("Warning: self.time_fe is None. Assembly y-independent terms terminated.")
            return
        M_t = self.time_fe.matrix["mass"]
        D_t = self.time_fe.matrix["derivative"]
        L_x = self.space_fe.matrix["laplace"]
        M_x = self.space_fe.matrix["mass"]
        self.matrix["L"] = scipy.sparse.kron(M_t, L_x)
        self.matrix["D_t"] = scipy.sparse.kron(D_t, M_x)
        self.matrix["M"] = scipy.sparse.kron(M_t, M_x)

    def assemble_W(self, W_t=None):
        "Given a value of the coefficient W_t (function of time, usually a Brownina motion), assemble operators in which W_t appears."
        if self.time_fe is None:
            if self.verbose:
                print("Warning: self.time_fe is None. Assembly y-dependent terms terminated.")
            return
        if W_t is None:
            if self.verbose:
                print("SpaceTimeFE Warning: W_T is None. Skip assembly W-dependent operators.")
            return
        self.time_fe.assemble_matrices_W(W_t)
        M_Wt = self.time_fe.matrix["W_mass"]
        M_W2t = self.time_fe.matrix["WW_mass"]
        self.matrix["M_W"] = scipy.sparse.kron(M_Wt, self.space_fe.matrix["mass"])
        self.matrix["M_WW"] = scipy.sparse.kron(M_W2t, self.space_fe.matrix["mass"])
