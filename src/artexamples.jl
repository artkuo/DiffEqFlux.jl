# https://diffeqflux.sciml.ai/stable/

using DiffEqSensitivity, OrdinaryDiffEq, Zygote

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())
loss(u0,p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1))
du01,dp1 = Zygote.gradient(loss,u0,p)

# this demonstrates that the machinery is already there to solve ODEs,
# compute AD gradients

# https://diffeqflux.sciml.ai/stable/examples/optimization_ode/

using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
using Plots
plot(sol)
savefig("LV_ode.png")

function loss(p)
  sol = solve(prob, Tsit5(), p=p, saveat = tsteps)
  loss = sum(abs2, sol.-1)
  return loss, sol
end

callback = function (p, l, pred)
  display(l)
  plt = plot(pred, ylim = (0, 6))
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

result_ode = DiffEqFlux.sciml_train(loss, p,
                                    ADAM(0.1),
                                    cb = callback,
                                    maxiters = 100)

# optimized parameters to keep output at 1

## Stochastic differential equation
# https://diffeqflux.sciml.ai/stable/examples/optimization_sde/

using DiffEqFlux, DifferentialEquations, Plots, Flux, Optim, DiffEqSensitivity
function lotka_volterra!(du,u,p,t)
  x,y = u
  α,β,γ,δ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = δ*x*y - γ*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)

function multiplicative_noise!(du,u,p,t)
  x,y = u
  du[1] = p[5]*x
  du[2] = p[6]*y
end
p = [1.5,1.0,3.0,1.0,0.3,0.3]

prob = SDEProblem(lotka_volterra!,multiplicative_noise!,u0,tspan,p)
sol = solve(prob)
plot(sol, ylim=(0,10))

prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol, ylim=(0,10))

using Statistics
ensembleprob = EnsembleProblem(prob)
@time sol = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=10_000)
truemean = mean(sol,dims=3)[:,:]
truevar  = var(sol,dims=3)[:,:]

function loss(p)
    tmp_prob = remake(prob,p=p)
    ensembleprob = EnsembleProblem(tmp_prob)
    tmp_sol = solve(ensembleprob,SOSRI(),saveat=0.1,trajectories=1000,sensealg=ForwardDiffSensitivity())
    arrsol = Array(tmp_sol)
    sum(abs2,truemean - mean(arrsol,dims=3)) + 0.1sum(abs2,truevar - var(arrsol,dims=3)),arrsol
  end
  
function cb2(p,l,arrsol)
    @show p,l
    means = mean(arrsol,dims=3)[:,:]
    vars = var(arrsol,dims=3)[:,:]
    p1 = plot(sol[1].t,means',lw=5)
    scatter!(p1,sol[1].t,truemean')
    p2 = plot(sol[1].t,vars',lw=5)
    scatter!(p2,sol[1].t,truevar')
    p = plot(p1,p2,layout = (2,1))
    display(p)
    false
end

pinit = [1.2,0.8,2.5,0.8,0.1,0.1]
@time res = DiffEqFlux.sciml_train(loss,pinit,ADAM(0.05),cb=cb2,maxiters = 100)

## Lotka-Volterra with Flux.train!
# https://diffeqflux.sciml.ai/stable/examples/lotka_volterra/#Lotka-Volterra-with-Flux.train!-1

using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test #using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [2.2, 1.0, 2.0, 0.4]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)

function predict_rd()
    Array(solve(prob,Tsit5(),saveat=0.1,reltol=1e-4))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())

# took 6327 scatter

## Optimal control
# https://diffeqflux.sciml.ai/stable/examples/optimal_control/#optcontrol-1

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Plots, Statistics, DiffEqSensitivity
tspan = (0.0f0,8.0f0)
ann = FastChain(FastDense(1,32,tanh), FastDense(32,32,tanh), FastDense(32,1))
θ = initial_params(ann)
function dxdt_(dx,x,p,t)
    x1, x2 = x
    dx[1] = x[2]                 # xdot
    dx[2] = ann([t],p)[1]^3      # vdot
end
x0 = [-4f0,0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_,x0,tspan,θ)
solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)
function predict_adjoint(θ)
  Array(solve(prob,Vern9(),p=θ,saveat=ts,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end
function loss_adjoint(θ)
  x = predict_adjoint(θ)
  mean(abs2,4.0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])/10
end
l = loss_adjoint(θ)
cb = function (θ,l)
  println(l)
  p = plot(solve(remake(prob,p=θ),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3, label=["x(t)" "v(t)"])
  plot!(p,ts,[first(ann([t],θ)) for t in ts],label="u(t)",lw=3)
  display(p)
  return false
end
# Display the ODE with the current parameter values.
cb(θ,l)
loss1 = loss_adjoint(θ)
# Train with ADAM a bit, then switch to BFGS
res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.005), cb = cb,maxiters=100)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.minimizer,
                              BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
                              allow_f_increases = false)

## Now refine with normal cost for control input

function loss_adjoint(θ)
    x = predict_adjoint(θ)
    mean(abs2,4.0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])
end
  
res3 = DiffEqFlux.sciml_train(loss_adjoint, res2.minimizer,
                                BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
                                allow_f_increases = false)
  
l = loss_adjoint(res3.minimizer)
cb(res3.minimizer,l)
p = plot(solve(remake(prob,p=res3.minimizer),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3)
plot!(p,ts,[first(ann([t],res3.minimizer)) for t in ts],label="u(t)",lw=3)
#savefig("optimal_control.png")

# I don't get the same answer they do, but the overall x(t) looks similar

## Bouncing ball
# https://diffeqflux.sciml.ai/stable/examples/bouncing_ball/#Bouncing-Ball-Hybrid-ODE-Optimization-1

using DiffEqFlux, Optim, OrdinaryDiffEq, DiffEqSensitivity

function f(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]
end

function condition(u,t,integrator) # Event when event_f(u,t) == 0
  u[1]
end

function affect!(integrator)
  integrator.u[2] = -integrator.p[2]*integrator.u[2]
end

cb = ContinuousCallback(condition,affect!)
u0 = [50.0,0.0]
tspan = (0.0,15.0)
p = [9.8, 0.8] # g and coefficient of restitution
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=range(tspan[1],stop=tspan[end], length=100), callback=cb)
plot(sol)

#

function loss(θ)
    sol = solve(prob,Tsit5(),p=[9.8,θ[1]],callback=cb,sensealg=ForwardDiffSensitivity())
    target = 20.0
    abs2(sol[end][1] - target), sol
end
  
loss([0.8])
res = DiffEqFlux.sciml_train(loss,[0.8],BFGS())
println("Final result coefficient of restitution = ", res.minimizer[1])
l, sol = loss(res.minimizer)
plot(sol)

# can't quite get to the desired height

