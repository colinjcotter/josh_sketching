from firedrake import *

ncells = 20
base_mesh = UnitIntervalMesh(ncells)

height = 1.
nlayers = 10
mesh = ExtrudedMesh(base_mesh, layer_height=height/nlayers)

# make some function spaces
CG1_1D_elt = FiniteElement("CG", interval, 1)
DG0_1D_elt = FiniteElement("DG", interval, 0)
Uelt = TensorProductElement(CG1_1D_elt, CG1_1D_elt)
VU = FunctionSpace(mesh, Uelt)
Pelt = TensorProductElement(CG1_1D_elt, DG0_1D_elt)
VP = FunctionSpace(mesh, Pelt)
W = VU * VP # the mixed space for the coupled x-p system

# solution!
w = Function(W)
u, p = split(w)

# model parameters
# initial condition from before seeing the data
x, t = SpatialCoordinate(mesh)
u0 = sin(2*pi*x)
# gamma - variance in initial condition error
gamma = Constant(0.1)
# variance in observation error
C = Constant(0.1)
# fake some data
u1 = 0.5*sin(2*pi*x)
# variance for model error
Sigma = Constant(0.1)

# the functional F

# penalty term for initial condition (f0)
F = 0.5*(u-u0)*(u-u0)/gamma*ds_b # integral over t=0 "surface"

# "data" (g0)
F += 0.5*(u-u1)*(u-u1)/C*ds_t # integral over t=1 "surface"

# "cost"
F += -0.5*p*p/Sigma*dx

# "dynamical constraint"
F += (p*u.dx(1) + p.dx(0)*u.dx(0))*dx

# the equation to solve
dF = derivative(F, w)

# make a problem to solve
daparams = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

DAProblem = NonlinearVariationalProblem(dF, w)
DASolver = NonlinearVariationalSolver(DAProblem,
                                      solver_parameters=daparams)
DASolver.solve()
