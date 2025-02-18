
using MultiComponentFlash
h2o = MolecularProperty(0.018015268, 22.064e6, 647.096, 5.595e-05, 0.3442920843)
co2 = MolecularProperty(0.0440098, 7.3773e6, 304.1282, 9.412e-05, 0.22394)

bic = [0 0;
       0 0]

mixture = MultiComponentMixture([h2o, co2], A_ij = bic, names = ["H2O", "CO2"])
eos = GenericCubicEOS(mixture, PengRobinson())

using Jutul, JutulDarcy
nx = 400
ny = 1
nz = 200
bar = 1e5
dims = (nx, ny, nz)
g = CartesianMesh(dims, (12.5, 10.0, 1.0) .* dims)
nc = number_of_cells(g)
Darcy = 9.869232667160130e-13
Kx = 1f-1 * ones(nx, nz) * Darcy
Kx[:, 100:105] .= 1f-4 * Darcy
K = vcat(vec(Kx)', vec(Kx)', vec(Kx)')
res = discretized_domain_tpfv_flow(tpfv_geometry(g), porosity = 0.3, permeability = K)
## Set up a vertical well in the first corner, perforated in top layer
prod = setup_well(g, K, [(nx, 1, nz)], name = :Producer)
## Set up an injector in the opposite corner, perforated in bottom layer
inj = setup_well(g, K, [(1, 1, nz)], name = :Injector)

rhoLS, rhoVS = 844.23, 126.97
rhoS = [rhoLS, rhoVS]
L, V = LiquidPhase(), VaporPhase()
# Define system and realize on grid
sys = MultiPhaseCompositionalSystemLV(eos, (L, V))
model, parameters = setup_reservoir_model(res, sys, wells = [inj, prod], reference_densities = rhoS, block_backend = true);
kr = BrooksCoreyRelPerm(sys, 2.0, 0.0, 1.0)
model = replace_variables!(model, RelativePermeabilities = kr)
T0 = repeat([303.15], 1, nc)
parameters[:Reservoir][:Temperature] = T0
state0 = setup_reservoir_state(model, Pressure = 50*bar, OverallMoleFractions = [1.0, 0.0]);


# 5 year (5*365.24 days)
day = 24*3600.0
dt = repeat([20]*day, 50)
rate_target = TotalRateTarget(5e-3)
I_ctrl = InjectorControl(rate_target, [0, 1], density = rhoVS)
bhp_target = BottomHolePressureTarget(50*bar)
P_ctrl = ProducerControl(bhp_target)

controls = Dict()
controls[:Injector] = I_ctrl
controls[:Producer] = P_ctrl
forces = setup_reservoir_forces(model, control = controls)

sim, config = setup_reservoir_simulator(model, state0, parameters, info_level = 1, max_timestep_cuts = 1000);
@time states, reports = simulate!(sim, dt, forces = forces, config = config);

## Once the simulation is done, we can plot the states

using PyPlot
matplotlib.use("agg")
for i = 1:length(states)
fig=figure();
subplot(1,2,1);
imshow(reshape(states[i][:Reservoir][:OverallMoleFractions][2,:], nx, nz)', vmin=0, vmax=maximum(states[end][:Reservoir][:OverallMoleFractions][2,:])); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(states[i][:Reservoir][:Pressure].-50*bar, nx, nz)', vmin=0, vmax=maximum(states[i][:Reservoir][:Pressure]).-50*bar); colorbar(); title("pressure - 50*bar")
savefig("plots/2D-layer/sat-p-$i.png", bbox_inches="tight", dpi=300);
close(fig)
end
