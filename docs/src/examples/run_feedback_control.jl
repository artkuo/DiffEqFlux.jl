

# diffeqflux stuff

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Plots

u0 = 1.1f0  # initial state of known system to be controlled
tspan = (0.0f0, 25.0f0)
tsteps = 0.0f0:1.0:25.0f0

# Neural model is a controller acting as differential equation
# with inputs model_control and system_output, and 
# output dmodel_control (to be integrated as an ODE)
model_univ = FastChain(FastDense(2, 16, tanh),
                       FastDense(16, 16, tanh),
                       FastDense(16, 1))

# The model weights are destructured into a vector of parameters
p_model = initial_params(model_univ)
n_weights = length(p_model)

# Parameters of the known system (second-order linear dynamics)
p_system = Float32[0.5, -0.5]

p_all = [p_model; p_system]
θ = Float32[u0; p_all]        # sciml will train both initial state and params
# θ = initial state of known system, NN weights, and known system coefs

##
function dudt_univ!(du, u, p, t)
    # Destructure the parameters
    model_weights = p[1:n_weights]
    α = p[end - 1] # system ODE's linear constant coefficients
    β = p[end]

    # The neural network outputs a control taken by the system
    # The system then produces an output
    model_control, system_output = u

    # Dynamics of the control and system
    dmodel_control = model_univ(u, model_weights)[1]
    dsystem_output = α*system_output + β*model_control

    # Update in place
    du[1] = dmodel_control
    du[2] = dsystem_output
end

##
# initial condition is a model control = 0, system state = u0
prob_univ = ODEProblem(dudt_univ!, [0f0, u0], tspan, p_all)
sol_univ = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6)

# 
# θ = initial state of known system, NN weights, and known system coefs
function predict_univ(θ)
  return Array(solve(prob_univ, Tsit5(), u0=[0f0, θ[1]], p=θ[2:end],
                              saveat = tsteps))
end

loss_univ(θ) = sum(abs2, predict_univ(θ)[2,:] .- 1)
l = loss_univ(θ)

##

list_plots = []
iter = 0
callback = function (θ, l)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  println(l)

  plt = plot(predict_univ(θ)', ylim = (0, 6))
  push!(list_plots, plt)
  display(plt)
  return false
end

##

result_univ = DiffEqFlux.sciml_train(loss_univ, θ,
                                     BFGS(initial_stepnorm = 0.01),
                                     cb = callback,
                                     allow_f_increases = false)


u0star = result_univ.minimizer[1]
pstar = result_univ.minimizer[2:n_weights+1]
αβstar = result_univ.minimizer[end-length(p_system)+1:end]
θstar = result_univ.minimizer
# I believe this system is finding an easy system to control,
# because it gets αstar = -61.7, βstar = 8.62, so that the
# plant dynamics are xdot = α x + β control. The system is
# super stable. I think it's not necessary to estimate alpha and
# beta, and we could ask the controller to really learn to control. 