## A simple 2D example for compass fluid-flow simulation

using Jutul
using JutulDarcy
using LinearAlgebra
using PyPlot
using SlimPlotting
using JLD2
using Images
JLD2.@load "data/compass/compass.jld2"

## grid size
x_true = imresize(x_true, 208, 84)
q_true = Int.(round.(q_true./2))
q_true[1] = 60
nx, nz = size(x_true)
ny = 1

## fluid info
sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()))
#ρCO2=7e2
#ρH20=1e3
ρCO2=501.9
ρH20=1053.0

## mesh, models
dims = (nx, ny, nz)
d = (24.0, 24.0, 24.0)
g = CartesianMesh(dims, d .* dims)
nc = number_of_cells(g)
Darcy = 9.869232667160130e-13
K = Float64.(vcat(vec(x_true)', vec(x_true)', vec(x_true)') * 1e-3 * Darcy)
G = discretized_domain_tpfv_flow(tpfv_geometry(g), porosity = 0.25, permeability = K)
nc = number_of_cells(G)

model = SimulationModel(G, sys)
kr = BrooksCoreyRelPerm(sys, [2.0, 2.0])
replace_variables!(model, RelativePermeabilities = kr)

## simulation time steppings
day = 24*3600.0
tstep = repeat([365]*day, 50)
tot_time = sum(tstep)

## injection & production
inj_loc = (q_true[1], 1, q_true[2])
prod_loc = (nx, 1, nz)
inj_cell = (inj_loc[end]-1)*nx+inj_loc[1]
prod_cell = (prod_loc[end]-1)*nx+prod_loc[1]
irate = 5e-3 * ρCO2
src  = [SourceTerm(inj_cell, irate, fractional_flow = [1.0, 0.0]), 
            SourceTerm(prod_cell, -irate, fractional_flow = [0.0, 1.0])]
#src  = [SourceTerm(inj_cell, irate, fractional_flow = [1.0, 0.0])]
forces = setup_forces(model, sources = src)  

## set up parameters
bar = 1e5
Z = reshape(repeat((1:nz)*d[3], outer = nx), nz, nx)'
p0 = vec(1000.0 * 10 * Z)

parameters = setup_parameters(model, PhaseViscosities = [1e-4, 1e-3], density = [ρCO2, ρH20]); # 0.1 and 1 cP
state0 = setup_state(model, Pressure = p0, Saturations = [0.0, 1.0]);

## simulation
@time states, _ = simulate(state0, model, tstep, parameters = parameters, forces = forces, info_level = 1, max_timestep_cuts = 1000, block_backend = false);

## plotting
using PyPlot
matplotlib.use("agg")
for i = 1:length(states)
fig=figure();
subplot(1,2,1);
imshow(reshape(states[i][:Saturations][1,:], nx, nz)', vmin=0, vmax=maximum(states[end][:Saturations][1,:])); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(states[i][:Pressure], nx, nz)', vmin=0, vmax=maximum(states[i][:Pressure])); colorbar(); title("pressure")
savefig("plots/compass/sat-p-$i.png", bbox_inches="tight", dpi=300);
close(fig)
end
