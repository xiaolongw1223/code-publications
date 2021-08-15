import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh

from SimPEG.utils import plot2Ddata, surface2ind_topo, model_builder
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)
from SimPEG.optimization import IterationPrinters, StoppingCriteria
from scipy.spatial import cKDTree

from time import time
from datetime import timedelta

# def run(plotIt=True):
init_t = time()
np.random.seed(0)

# Read in mesh
mesh = TensorMesh._readUBC_3DMesh("mesh.txt")

# Read data 
data_object = utils.io_utils.readUBCgravityObservations("grav.obs")
dobs = -data_object.dobs
uncertainties = data_object.standard_deviation
receiver_locations = data_object.survey.receiver_locations

# Load topography
xyz_topo = np.genfromtxt("topo.topo", skip_header=1)


# Define the receivers. The data consist of vertical gravity anomaly measurements.
# The set of receivers must be defined as a list.
receiver_list = gravity.receivers.Point(receiver_locations, components="gzz")

receiver_list = [receiver_list]

# Define the source field
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Define the survey
survey = gravity.survey.Survey(source_field)

#############################################
# Defining the Data
# -----------------
#
# Here is where we define the data that are inverted. The data are defined by
# the survey, the observation values and the standard deviation.
#

data_object = data.Data(survey, dobs=dobs, standard_deviation=uncertainties)


# Define density contrast values for each unit in g/cc. Don't make this 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.
background_density = 1e-6

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Define and plot starting model
starting_model = background_density * np.ones(nC)


# Here, we define the physics of the gravity problem by using the simulation
# class.
#

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh, rhoMap=model_map, actInd=ind_active
)


# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)
dmis.W = utils.sdiag(1 / uncertainties)

# depth weighting
threshold = 2
exponent = 3
tree = cKDTree(xyz_topo[:, :-1])
_, ind = tree.query(mesh.cell_centers[:, :-1])
delta_z = np.abs(mesh.cell_centers[:, -1] - xyz_topo[ind, -1])
wz = (delta_z + threshold) ** (-0.5 * exponent)
wz = wz * mesh.vol
wz = wz[ind_active]
wr = wz / np.nanmax(wz)

# Define the regularization (model objective function).
reg = regularization.Sparse(mesh, indActive=ind_active, mapping=model_map)
reg.alpha_s = 4.58/5
reg.norms = np.c_[0.03, 2, 2, 2]
reg.cell_weights = wr


printers = [
    IterationPrinters.iteration, IterationPrinters.beta,
    IterationPrinters.phi_d, IterationPrinters.phi_m, 
    IterationPrinters.f, IterationPrinters.iterationCG,
    IterationPrinters.totalLS, IterationPrinters.comment
    ]
# stoppers = [StoppingCriteria.iteration, StoppingCriteria.moving_x]

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=500, lower=-5.0, upper=5.0, maxIterLS=20, maxIterCG=5000, tolCG=1e-3
)
opt.printers = printers 
# opt.stoppers = stoppers

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e-1)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-3, max_irls_iterations=300, beta_tol=1e-2,
)

# Defining the fractional decrease in beta and the number of Gauss-Newton solves
# for each beta value.
beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=True)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Add sensitivity weights
# sensitivity_weights = directives.UpdateSensitivityWeights(everyIter=False)

# The directives are defined as a list.
directives_list = [
    update_IRLS,
    # sensitivity_weights,
    starting_beta,
    beta_schedule,
    save_iteration,
    update_jacobi,
]


# Here we combine the inverse problem and the set of directives
inv = inversion.BaseInversion(inv_prob, directives_list)

# Run inversion
recovered_model = inv.run(starting_model)

plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

mesh.writeModelUBC('model_rec.txt', plotting_map*recovered_model)
np.savetxt('dpred.txt', inv_prob.dpred)
np.savetxt('dnres.txt', (dobs-inv_prob.dpred) / uncertainties)

elapse = time() - init_t
print("elapsed time: ", str(timedelta(seconds=elapse)))