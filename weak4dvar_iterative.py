from firedrake import *

ncells = 50
base_mesh = UnitIntervalMesh(ncells)

height = 1.
nlayers = 50
mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=height/nlayers)

# make some function spaces
CG1_1D_elt = FiniteElement("CG", interval, 1)
DG0_1D_elt = FiniteElement("DG", interval, 0)
Uelt = TensorProductElement(CG1_1D_elt, CG1_1D_elt)
VU = FunctionSpace(mesh, Uelt)
Pelt = TensorProductElement(CG1_1D_elt, DG0_1D_elt)
VP = FunctionSpace(mesh, Pelt)
W = VU * VP # the mixed space for the coupled x-p system

# solution!
wkp1 = Function(W)
ukp1, pkp1 = split(wkp1)
wk = Function(W)
uk, pk = split(wk)

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

# "data" (g0)
F = 0.5*(ukp1-u1)*(ukp1-u1)/C*ds_t # integral over t=1 "surface"

# "cost"
F += -0.5*pkp1*pkp1/Sigma*dx

# "dynamical constraint"
F += (pkp1*ukp1.dx(1) + pkp1.dx(0)*ukp1.dx(0))*dx
# penalty term for initial condition (f0)
F += 0.5*(ukp1-u0)*(ukp1-u0)/gamma*ds_b # integral over t=0 "surface"

# the equation to solve
dF = derivative(F, wkp1)

# preconditioning operator
# "data" (g0)
#Fp = 0.5*(ukp1-u1)*(ukp1-u1)/C*ds_t # integral over t=1 "surface"
# "cost"
Fp = -0.5*pkp1*pkp1/Sigma*dx
# "dynamical constraint"
Fp += (pkp1*ukp1.dx(1) + pkp1.dx(0)*ukp1.dx(0))*dx

JFp = derivative(derivative(Fp, wkp1), wkp1)

bcs = [DirichletBC(W.sub(0), u0 - gamma*pk, "bottom")]

class DiagPC(PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "Diag_"

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = context.appctx
        self.appctx = appctx

        # FunctionSpace checks
        u, v = context.a.arguments()
        if u.function_space() != v.function_space():
            raise ValueError("Pressure space test and trial space differ")

        V = u.function_space()
        self.xfstar = Cofunction(V.dual())
        self.yf = Function(V)
        
    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
            
        # THIS IS WHERE WE DO SOMETHING
            
        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

# make a problem to solve
lu_params = {
    "snes_type": "ksponly",
    "snes_monitor": None,
    "ksp_type": "gmres",
    "ksp_atol": 1.0e-50,
    "ksp_rtol": 1.0e-8,
    "ksp_monitor": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
    
DAProblem = NonlinearVariationalProblem(dF, wkp1, Jp=JFp, bcs=bcs)
DASolver = NonlinearVariationalSolver(DAProblem,
                                      solver_parameters=lu_params)

err = 1.0e50
tol = 1.0e-6
while err > tol:
    DASolver.solve()
    err = errornorm(wk, wkp1)
    print(err)
    wk.assign(wkp1)
    
uout, pout = wkp1.subfunctions
f0 = File("test4dvar.pvd")
f0.write(uout, pout)
