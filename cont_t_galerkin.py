"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
import scipy.sparse
from ufl import TestFunction, TrialFunction, dx, grad, inner

from utils import float_f
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def cart_prod_coords(t_coords, x_coords):
    if len(x_coords.shape) == 1:
        x_coords = np.expand_dims(x_coords, 1)
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


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


def assemble_ctg_slab(Space, u0, Time, exact_rhs, boundary_data):
    # Assemble space-time matrices (linear PDEs -> Kronecker product t & x matrices)
    mass_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["mass"])
    stiffness_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["laplace"])
    system_matrix = (
        scipy.sparse.kron(Time.matrix["derivative"], Space.matrix["mass"])
        + stiffness_matrix
    )

    # Assemble RHS vector as space-time mass * RHS on dofs
    # TODO better to use projection? I'll use higher order FEM!
    space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
    rhs = mass_matrix.dot(exact_rhs(space_time_coords))

    # Impose initial condition strongly
    dofs_at_t0 = np.zeros((Time.n_dofs * Space.n_dofs))  # indicator dofs at t_0
    dofs_at_t0[: Space.n_dofs] = 1.0

    system_matrix = system_matrix.multiply((1.0 - dofs_at_t0).reshape(-1, 1))
    system_matrix += scipy.sparse.diags(dofs_at_t0)
    rhs[: Space.n_dofs] = u0

    # Impose boundary conditions
    # Idea: modify A and RHS so that, if the i-th dof belongs to the boundary, then the i-th equation enforces the BC instead of the equation wrt the i-th test function. This means that:
    # * the i-th row of A becomes delta_{i,j}
    # * the i-th entry of RHS becomes i-th coordinate of BC

    # 1. recover data
    dofs_boundary = np.kron(
        np.ones((Time.dofs.shape[0], 1)), Space.boundary_dof_vector.reshape(-1, 1)
    ).flatten()
    bc_curr_slab = boundary_data(space_time_coords)

    # 2. Edit system matrix
    system_matrix = system_matrix.multiply(
        (1.0 - dofs_boundary).reshape(-1, 1)
    )  # put to 0 entries corresponding to boundary
    system_matrix += scipy.sparse.diags(dofs_boundary)

    # 3. Edit RHS vector
    rhs = rhs * (1.0 - dofs_boundary)
    rhs += bc_curr_slab * dofs_boundary

    return system_matrix, mass_matrix, stiffness_matrix, rhs, bc_curr_slab


def run_CTG_elliptic(
    comm,
    Space,
    n_time,
    order_t,
    time_slabs,
    boundary_data,
    exact_rhs,
    initial_data,
    exact_sol=None,
    err_type="h1",
    verbose=False,
):
    # coordinates initial condition wrt space-time basis
    init_time = time_slabs[0][0]
    u0 = initial_data(cart_prod_coords(np.array([[init_time]]), Space.dofs))

    total_n_dofs_t = 0
    sol_slabs = []
    err_slabs = np.zeros((len(time_slabs),))  # square L2 error current slab
    norm_u_slabs = np.zeros_like(err_slabs)  # square L2 norm apx. sol.

    # Time marching over slabs
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(
                f"Solving on slab_{i} = D x ({round(slab[0], 5)}, {round(slab[1], 5)}) ...",
                flush=True,
            )

        # Compute FE object for current slab TIME discretization
        msh_t = mesh.create_interval(comm, n_time, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        Time = TimeFE(msh_t, V_t)
        total_n_dofs_t += Time.n_dofs

        # Assemble linear system
        system_matrix, mass_matrix, stiffness_matrix, rhs, ex_sol_slab = (
            assemble_ctg_slab(Space, u0, Time, exact_rhs, boundary_data)
        )

        # Solve linear system (sparse direct solver)
        sol_slab = scipy.sparse.linalg.spsolve(system_matrix, rhs)

        # Check residual
        residual_slab = system_matrix.dot(sol_slab) - rhs
        rel_res_slab = np.linalg.norm(residual_slab) / np.linalg.norm(sol_slab)
        warn = False
        if rel_res_slab > 1.0e-4:
            warn = True
            print("WARNING: ", end="")
        if verbose or warn:
            print(f"Relative residual solver slab {i}:", float_f(rel_res_slab))

        # Get initial condition on next slab = final condition from this slab
        last_time_dof = Time.dofs.argmax()
        u0 = sol_slab[last_time_dof * Space.n_dofs : (last_time_dof + 1) * Space.n_dofs]

        # Error curr slab
        if callable(exact_sol):  # compute error only if exact_sol is a function
            err_slabs[i], norm_u_slabs[i] = compute_error_slab(
                Space,
                exact_sol,
                err_type,
                Time,
                sol_slab,
                # mass_matrix,
                # stiffness_matrix
            )

            if verbose:
                print("Current L2 error", float_f(err_slabs[i]))
                print(
                    "Current L2 relative error", float_f(err_slabs[i] / norm_u_slabs[i])
                )
                print("Done.\n")

    n_dofs = Space.n_dofs * total_n_dofs_t
    return err_slabs, norm_u_slabs, n_dofs


def compute_error_slab(
    Space, exact_sol, err_type, Time, sol_slab
):  # mass_mat, stif_mat, ):
    # --------------------------------- WRONG -------------------------------- #
    # space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
    # ex_sol_slab = exact_sol(space_time_coords)  
    # # compute exact sol on dofs; i.e. PROJECT exact sol in discrete sapce
    # --------------------------------- WRONG -------------------------------- #

    # refine Time
    msh_t = Time.mesh
    msh_t_ref = mesh.refine(msh_t)[0]
    p_Time = Time.V.element.basix_element.degree
    V_t_ref = fem.functionspace(msh_t_ref, ("Lagrange", p_Time))
    Time_ref = TimeFE(msh_t_ref, V_t_ref)

    # refine Space
    msh_x = Space.mesh
    msh_x_ref = mesh.refine(msh_x)[0]
    p_Space = Space.V.element.basix_element.degree
    V_x_ref = fem.functionspace(msh_x_ref, ("Lagrange", p_Space))
    Space_ref = SpaceFE(msh_x_ref, V_x_ref)

    # Interpolate exact sol in fine space # TODO valid only for P=1!
    fine_coords = cart_prod_coords(Time_ref.dofs, Space_ref.dofs)
    ex_sol_ref = exact_sol(fine_coords)

    # Interpolate using griddata (linear interpolation) # TODO valid only for P=1!
    coarse_coords = cart_prod_coords(Time.dofs, Space.dofs)
    sol_slab_ref = griddata(
        coarse_coords, sol_slab, fine_coords, method="linear", fill_value=0.0
    )

    # Compute IP matrix
    mass_matrix = scipy.sparse.kron(Time_ref.matrix["mass"], Space_ref.matrix["mass"])
    if err_type == "h1":
        stiffness_matrix = scipy.sparse.kron(
            Time_ref.matrix["mass"], Space_ref.matrix["laplace"]
        )
        ip_matrix = mass_matrix + stiffness_matrix
    elif err_type == "l2":
        ip_matrix = mass_matrix
    else:
        raise ValueError(f"Unknown error type: {err_type}")

    err_fun_ref = ex_sol_ref - sol_slab_ref
    err = sqrt(ip_matrix.dot(err_fun_ref).dot(err_fun_ref))
    norm_u = sqrt(ip_matrix.dot(sol_slab_ref).dot(sol_slab_ref))
    return err, norm_u
