## A simple 2D example for fluid-flow simulation

using Jutul
using JutulDarcy
using LinearAlgebra
using PyPlot
using SlimPlotting

## grid size
nx = 30
ny = 1
nz = 15

## fluid info
sys = ImmiscibleSystem((LiquidPhase(), VaporPhase()))
#ρCO2=7e2
#ρH20=1e3
visCO2 = 1e-4
visH2O = 1e-3
ρCO2=501.9
ρH2O=1053.0

## mesh, models
dims = (nx, ny, nz)
d = (30.0, 30.0, 30.0)
g = CartesianMesh(dims, d .* dims)
nc = number_of_cells(g)
Darcy = 9.869232667160130e-13
Kx = 0.02 * ones(nx, nz) * Darcy
Kxtrue = deepcopy(Kx)
Kxtrue[:,8:10] .*= 6.0
K = vcat(vec(Kx)', vec(Kx)', vec(Kx)')
Ktrue = vcat(vec(Kxtrue)', vec(Kxtrue)', vec(Kxtrue)')
G = discretized_domain_tpfv_flow(tpfv_geometry(g), porosity = 0.25, permeability = K)
Gtrue = discretized_domain_tpfv_flow(tpfv_geometry(g), porosity = 0.25, permeability = Ktrue)
nc = number_of_cells(G)

model = SimulationModel(G, sys)
modeltrue = SimulationModel(Gtrue, sys)
kr = BrooksCoreyRelPerm(sys, [2.0, 2.0])
replace_variables!(model, RelativePermeabilities = kr)
modeltrue = SimulationModel(Gtrue, sys)

## simulation time steppings
day = 24*3600.0
tstep = repeat([20]*day, 50)
tot_time = sum(tstep)

## injection & production
inj_loc = (3, 1, 9)
prod_loc = (28, 1, 9)
inj_cell = (inj_loc[end]-1)*nx+inj_loc[1]
prod_cell = (prod_loc[end]-1)*nx+prod_loc[1]
irate = 5e-3 * ρCO2
src  = [SourceTerm(inj_cell, irate, fractional_flow = [1.0, 0.0]), 
            SourceTerm(prod_cell, -irate, fractional_flow = [0.0, 1.0])]
forces = setup_forces(model, sources = src)  

## set up parameters
bar = 1e5
Z = reshape(repeat((1:nz)*d[3], outer = nx), nz, nx)'
p0 = vec(1000.0 * 10 * Z)

parameters = setup_parameters(model, PhaseViscosities = [visCO2, visH2O], density = [ρCO2, ρH2O]); # 0.1 and 1 cP
state0 = setup_state(model, Pressure = p0, Saturations = [0.0, 1.0]);

## simulation
states, _ = simulate(state0, model, tstep, parameters = parameters, forces = forces, info_level = 1, max_timestep_cuts = 1000);
statestrue, _ = simulate(state0, modeltrue, tstep, parameters = parameters, forces = forces, info_level = 1, max_timestep_cuts = 1000);

## plotting
using PyPlot
matplotlib.use("agg")
for i = 1:length(states)
fig=figure();
subplot(1,2,1);
imshow(reshape(states[i][:Saturations][1,:], nx, nz)', vmin=0, vmax=maximum(states[end][:Saturations][1,:])); colorbar(); title("saturation")
subplot(1,2,2);
imshow(reshape(states[i][:Pressure], nx, nz)', vmin=0, vmax=maximum(states[i][:Pressure])); colorbar(); title("pressure")
savefig("plots/basic2D/sat-p-$i.png", bbox_inches="tight", dpi=300);
close(fig)
end

h = 30.0
dt = 20
figure(figsize=(20,12))
for i = 1:6
    subplot(2,3,i)
    if i == 1
        plot_velocity(reshape(state0[:Saturations][1,:], nx, nz)', (h, h); new_fig=false, vmax=1)
    else
        plot_velocity(reshape(states[10*(i-1)][:Saturations][1,:], nx, nz)', (h, h); new_fig=false, vmax=1)
    end
    colorbar()
    title("CO2 concentration at day $((i-1)*10*dt)")
end
savefig("plot-it-same-s.png", dpi=300, bbox_inches="tight")

figure(figsize=(20,12))
for i = 1:6
    subplot(2,3,i)
    if i == 1
        plot_velocity(reshape(state0[:Pressure], nx, nz)', (h, h); new_fig=false)
    else
        plot_velocity(reshape(states[10*(i-1)][:Pressure], nx, nz)', (h, h); new_fig=false)
    end
    colorbar()
    title("pressure at day $((i-1)*10*dt)")
end
savefig("plot-it-same-p.png", dpi=300, bbox_inches="tight")
