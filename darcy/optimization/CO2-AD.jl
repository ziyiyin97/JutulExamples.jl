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
ρCO2=501.9
ρH20=1053.0

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
replace_variables!(modeltrue, RelativePermeabilities = kr)

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

parameters = setup_parameters(model, PhaseViscosities = [1e-4, 1e-3], density = [ρCO2, ρH20]); # 0.1 and 1 cP
parameterstrue = setup_parameters(modeltrue, PhaseViscosities = [1e-4, 1e-3], density = [ρCO2, ρH20]); # 0.1 and 1 cP
state0 = setup_state(model, Pressure = p0, Saturations = [0.0, 1.0]);

## simulation
states, _ = simulate(state0, model, tstep, parameters = parameters, forces = forces, info_level = 1, max_timestep_cuts = 1000);
statestrue, _ = simulate(state0, modeltrue, tstep, parameters = parameterstrue, forces = forces, info_level = 1, max_timestep_cuts = 1000);

# Define objective
function mass_mismatch(model, state, dt, step_no, forces)
    state_ref = statestrue[step_no]
    fld = :Saturations
    val = state[fld]
    ref = state_ref[fld]
    err = 0
    for i in axes(val, 2)
        err += (val[1, i] - ref[1, i])^2
    end
    return dt*err
end

@assert Jutul.evaluate_objective(mass_mismatch, model, statestrue, tstep, forces) == 0.0
@assert Jutul.evaluate_objective(mass_mismatch, model, states, tstep, forces) > 0.0

## Set up a configuration for the optimization
cfg = optimization_config(model, parameters, use_scaling = true, rel_min = 0.1, rel_max = 10)
for (ki, vi) in cfg
    if ki in [:TwoPointGravityDifference,
              :PhaseViscosities]
        # We are not interested in matching gravity effects or viscosity here.
        vi[:active] = false
    end
    if ki == :Transmissibilities
        # Transmissibilities are derived from permeability and varies significantly. We can set
        # log scaling to get a better conditioned optimization system, without changing the limits
        # or the result.
        vi[:scaler] = :log
    end
end
print_obj = 100
# Set up parameter optimization
F_o, dF_o, F_and_dF, x0, lims, data = setup_parameter_optimization(model, state0, parameters, tstep, forces, mass_mismatch, cfg, print = print_obj, param_obj = true);
F_initial = F_o(x0)
dF_initial = dF_o(similar(x0), x0)

@info "Initial objective: $F_initial, gradient norm $(norm(dF_initial))"
## Link to an optimizer package
# We use Optim.jl but the interface is general enough that e.g. LBFGSB.jl can
# easily be swapped in.
#
# LBFGS is a good choice for this problem, as Jutul provides sensitivities via
# adjoints that are inexpensive to compute.
using Optim
lower, upper = lims
inner_optimizer = LBFGS()
opts = Optim.Options(store_trace = true, show_trace = true, time_limit = 2000)
results = optimize(Optim.only_fg!(F_and_dF), lower, upper, x0, Fminbox(inner_optimizer), opts)
x = results.minimizer
x_truth = vectorize_variables(modeltrue, parameterstrue, data[:mapper], config = data[:config])
F_final = F_o(x)

## Compute the solution using the tuned parameters found in x.
parameters_t = deepcopy(parameters)
devectorize_variables!(parameters_t, model, x, data[:mapper], config = data[:config])

states_tuned, = simulate(state0, model, tstep, parameters = parameters_t, forces = forces, info_level = -1);

@info "Final residual $F_final (down from $F_initial)"
fig = figure();
title("Scaled parameters")
plot(x, label="Final X")
plot(x0, label="Initial X")
plot(x_truth, label="True X")
legend()
savefig("final-x.png", bbox_inches="tight", dpi=300)

## Plot the final solutions.
# Note that we only match saturations - so any match in pressure
# is not guaranteed.
fig = figure();
subplot(3,2,1)
plot_velocity(reshape(statestrue[end][:Saturations][1, :], nx, nz)', (3f1, 3f1); new_fig=false, name = "ground truth", vmax=1); colorbar();
subplot(3,2,3)
plot_velocity(reshape(states[end][:Saturations][1, :], nx, nz)', (3f1, 3f1); new_fig=false, name = "initial guess", vmax=1); colorbar();
subplot(3,2,5)
plot_velocity(reshape(states_tuned[end][:Saturations][1, :], nx, nz)', (3f1, 3f1); new_fig=false, name = "inverted", vmax=1); colorbar();
subplot(3,2,2)
plot_velocity(reshape(statestrue[end][:Pressure], nx, nz)', (3f1, 3f1); new_fig=false, name = "ground truth"); colorbar();
subplot(3,2,4)
plot_velocity(reshape(states[end][:Pressure], nx, nz)', (3f1, 3f1); new_fig=false, name = "initial guess"); colorbar();
subplot(3,2,6)
plot_velocity(reshape(states_tuned[end][:Pressure], nx, nz)', (3f1, 3f1); new_fig=false, name = "inverted"); colorbar();
savefig("final-s-p.png", bbox_inches="tight", dpi=300)

## Plot the objective history and function evaluations
fig = figure();
subplot(1,2,1)
plot(data[:obj_hist][2:end])
title("Objective evaluations")
xlabel("Iterations")
ylabel("Objective")
subplot(1,2,2)
plot(map(x -> x.value, Optim.trace(results)))
title("Outer optimizer")
xlabel("Iterations")
ylabel("Objective")
savefig("iter.png", bbox_inches="tight", dpi=300)

