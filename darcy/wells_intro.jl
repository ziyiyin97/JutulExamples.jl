using JutulDarcy, Jutul
## Define and plot the mesh
nx = 20
ny = 10
nz = 4
# Some useful constants
day = 3600*24
bar = 1e5
# Create and plot the mesh
dims = (nx, ny, nz)
g = CartesianMesh(dims, (2000.0, 1500.0, 50.0))
plot_mesh(g)
## Create a layered permeability field
Darcy = 9.869232667160130e-13
nlayer = nx*ny
K = vcat(repeat([0.65], nlayer), repeat([0.3], nlayer), repeat([0.5], nlayer), repeat([0.2], nlayer))*Darcy
## Set up a vertical well in the first corner, perforated in all layers
P = setup_vertical_well(g, K, 1, 1, name = :Producer);
## Set up an injector in the upper left corner
I = setup_well(g, K, [(nx, ny, 1)], name = :Injector);

## Set up a two-phase immiscible system and define a density secondary variable
phases = (LiquidPhase(), VaporPhase())
rhoLS = 1000.0
rhoGS = 100.0
rhoS = [rhoLS, rhoGS]
sys = ImmiscibleSystem(phases, reference_densities = rhoS)
c = [1e-6/bar, 1e-4/bar]
ρ = ConstantCompressibilityDensities(p_ref = 1*bar, density_ref = rhoS, compressibility = c)
## Set up a reservoir model that contains the reservoir, wells and a facility that controls the wells
model, parameters = setup_reservoir_model(g, sys, wells = [I, P], backend = :csc, block_backend = true)
display(model)
## Replace the density function with our custom version
replace_variables!(model, PhaseMassDensities = ρ)
## Set up initial state
state0 = setup_reservoir_state(model, Pressure = 150*bar, Saturations = [1.0, 0.0])
## Set up time-steps
dt = repeat([30.0]*day, 12*5)
## Inject a full pore-volume (at reference conditions) of gas
# We first define an injection rate
reservoir = reservoir_model(model);
pv = pore_volume(model)
inj_rate = sum(pv)/sum(dt)
# We then set up a total rate target (positive value for injection)
# together with a corresponding injection control that specifies the
# mass fractions of the two components/phases for pure gas injection,
# with surface density given by the known gas density.
rate_target = TotalRateTarget(inj_rate)
I_ctrl = InjectorControl(rate_target, [0.0, 1.0], density = rhoGS)
# The producer operates at a fixed bottom hole pressure
bhp_target = BottomHolePressureTarget(50*bar)
P_ctrl = ProducerControl(bhp_target)
# Set up the controls. One control per well in the Facility.
controls = Dict()
controls[:Injector] = I_ctrl
controls[:Producer] = P_ctrl
# Set up forces for the whole model. For this example, all forces are defaulted
# (amounting to no-flow for the reservoir).
forces = setup_reservoir_forces(model, control = controls)
## Finally simulate!
sim, config = setup_reservoir_simulator(model, state0, parameters, info_level = 0)
@time states, reports = simulate!(sim, dt, forces = forces, config = config);
@time states, reports = simulate!(sim, dt, forces = forces, config = config);

function mass_mismatch(m, state, dt, step_no, forces)
       state_ref = states[step_no]
       fld = :Saturations
       val = state[:Reservoir][fld]
       ref = state_ref[:Reservoir][fld]
       err = 0
       for i in axes(val, 2)
           err += (val[1, i] - ref[1, i])^2
       end
       return dt*err
end
@assert Jutul.evaluate_objective(mass_mismatch, model, states, dt, forces) == 0.0
cfg = optimization_config(model, parameters, use_scaling = true, rel_min = 0.1, rel_max = 10)
for (ki, vi) in cfg[:Reservoir]
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
for keywords in [:Injector, :Producer, :Facility]
    for (ki, vi) in cfg[keywords]
        vi[:active] = false
    end
end

print_obj = 100

F_o, dF_o, F_and_dF, x0, lims, data = setup_parameter_optimization(model, state0, parameters, dt, forces, mass_mismatch, cfg, print = print_obj, param_obj = true); # this errors
