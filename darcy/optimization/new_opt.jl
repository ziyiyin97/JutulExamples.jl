using JutulDarcy, Jutul, LinearAlgebra
using Random
Random.seed!(2023)
nx = 20
ny = 10
nz = 4

initial_poro = 0.1
day = 3600*24
bar = 1e5
Darcy = 9.869232667160130e-13

dims = (nx, ny, nz)
g = CartesianMesh(dims, (2000.0, 1500.0, 100.0))
geo = tpfv_geometry(g)
pvol = sum(geo.volumes)*initial_poro
total_time = 12*5*day # ~5 years of injection
inj_rate = 0.025*pvol/total_time

##
function simple_co2_setup(g = g, poro = initial_poro, K = 0.1*Darcy, irate = inj_rate; time = total_time, nstep = 12*5)
    K = repeat([K], 1, number_of_cells(g))
    I = setup_well(g, K, [(nx, ny, 1)], name = :Injector);

    ## Set up a two-phase immiscible system and define a density secondary variable
    phases = (LiquidPhase(), VaporPhase())
    rhoWater = 1000.0
    rhoCO2 = 730.0
    rhoS = [rhoWater, rhoCO2]
    sys = ImmiscibleSystem(phases, reference_densities = rhoS)
    c = [1e-6/bar, 1e-4/bar]
    domain = discretized_domain_tpfv_flow(tpfv_geometry(g), porosity = poro, permeability = K)
    ρ = ConstantCompressibilityDensities(p_ref = 150*bar, density_ref = rhoS, compressibility = c)
    model, parameters = setup_reservoir_model(domain, sys, wells = [I])
    Jutul.select_output_variables!(model.models.Reservoir, :all)
    replace_variables!(model, PhaseMassDensities = ρ)
    replace_variables!(model, PhaseViscosities = vcat(1e-3 * ones(number_of_cells(g))', 1e-4 * ones(number_of_cells(g))'))
    for x ∈ keys(model.models)
        Jutul.select_output_variables!(model.models[x], :all)
    end
    state0 = setup_reservoir_state(model, Pressure = 150*bar, Saturations = [1.0, 0.0])
    dt = repeat([time/nstep], nstep)
    rate_target = TotalRateTarget(irate)
    I_ctrl = InjectorControl(rate_target, [0.0, 1.0], density = rhoCO2)
    controls = Dict()
    controls[:Injector] = I_ctrl
    forces = setup_reservoir_forces(model, control = controls)
    case = JutulCase(model, dt, forces; state0 = state0, parameters = parameters)
    return (model, dt, forces, state0, parameters)
end

model_ref, dt, forces, state0, parameters_ref = simple_co2_setup()
sim, config = setup_reservoir_simulator(model_ref, state0, parameters_ref);
@time states_ref, reports = simulate!(sim, dt, forces = forces, config = config, info_level=1);
##

function mass_mismatch(m, state, dt, step_no, forces)
    state_ref = states_ref[step_no]
    fld = :Saturations
    fld2 = :Pressure
    val = state[:Reservoir][fld]
    ref = state_ref[:Reservoir][fld]
    val2 = state[:Reservoir][fld2]
    ref2 = state_ref[:Reservoir][fld2]
    return 0.5 * sum((val[1,:] - ref[1,:]).^2) + 0.5 * sum((val2-ref2).^2)
end
##
# Perturbed porosity
model, dt, forces, state0, parameters = simple_co2_setup(g, 0.25)
sim, config = setup_reservoir_simulator(model, state0, parameters);
@time states, reports = simulate!(sim, dt, forces = forces, config = config, info_level=1);

@assert Jutul.evaluate_objective(mass_mismatch, model, states_ref, dt, forces) == 0.0
@assert Jutul.evaluate_objective(mass_mismatch, model, states, dt, forces) > 0.0

##
opt_config = optimization_config(model, parameters, Dict(:Reservoir => [:FluidVolume, :Transmissibilities], :Injector => [:FluidVolume]))
opt_config[:Reservoir][:Transmissibilities][:scaler] = :log
F_o, dF_o, F_and_dF, x0, lims, data = setup_parameter_optimization(
    model, state0, parameters, dt, forces, mass_mismatch, opt_config, param_obj = true, print = 0, config = config);

##
@time F_initial = F_o(x0)
@time dF_initial = dF_o(similar(x0), x0)
@info "Initial objective: $F_initial, gradient norm $(norm(dF_initial))"

mean(x) = sum(x)/length(x)

using Printf, Test
function grad_test(misfit, x0, dx, g; maxiter=6, h0=5f-2, data=false, stol=1f-1)
    # init
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    
    gdx = data ? g : dot(g, dx)
    f0 = misfit(x0)
    h = h0

    @printf("%11.5s, %11.5s, %11.5s, %11.5s, %11.5s, %11.5s \n", "h", "gdx", "e1", "e2", "rate1", "rate2")
    for j=1:maxiter
        f = misfit(x0 + h*dx)
        err1[j] = norm(f - f0, 1)
        err2[j] = norm(f - f0 - h*gdx, 1)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("%5.5e, %5.5e, %5.5e, %5.5e, %5.5e, %5.5e \n", h, h*norm(gdx, 1), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
        h = h * .8f0
    end

    rate1 = err1[1:end-1]./err1[2:end]
    rate2 = err2[1:end-1]./err2[2:end]
    @test isapprox(mean(rate1), 1.25f0; atol=stol)
    @test isapprox(mean(rate2), 1.5625f0; atol=stol)
end

dx0 = randn(length(x0))
dx0 = dx0/norm(dx0) * norm(x0)/5e5
grad_test(F_o, x0, dx0, dF_initial)

f(x0_trans) = vcat(x0_trans, x0[2081:end])
x0_trans = x0[1:2080]
dF_initial_trans = dF_initial[1:2080]
dx0_trans = dx0[1:2080]

grad_test(F_o ∘ f, x0_trans, dx0_trans, dF_initial_trans)
