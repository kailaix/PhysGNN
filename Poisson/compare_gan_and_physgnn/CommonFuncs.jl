using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using DelimitedFiles
using Distributions
Random.seed!(233)
matplotlib.use("agg")

solve_tridiagonal = load_op_and_grad("../ThomasOp/build/libSolveTridiagonal", "solve_tridiagonal")

# generating samples from Dirichlet distribution
dist_id = 1
if length(ARGS)>0
dist_id = parse(Int64, ARGS[1])
end
if dist_id==1
    global dist0 = Dirichlet(3, 1.0)
    global vmax = 5
else
    global dist0 = Dirichlet([1.0;2.0;3.0])
    global vmax = 12
end
# dist0 = MvNormal([0.5;0.5], [0.1 -0.05;-0.05 0.1])
# dist0 = MvNormal([0.5;0.5], 0.5*[1.0 0.5;0.5 2.0])
function get_sample(n)
    V = rand(dist0, n)
    V[1,:], V[2,:]
end

# Computing the coefficient
function avals(x,μ,σ)
    return 1 - 0.9exp(-(x-μ)^2/(2σ^2))
end

# #discretization intervals
n = 100

function PDE(g)
    @assert size(g, 2)==2
    us = Array{Any}(undef, 32) # batch size = 32
    for i = 1:32
        s = g[i]
        ax = constant(collect((1:2:2n+1)/2*1/(n+1)))
        aval = avals(ax, s[1], s[2])

        h = 1/(n+1)
        b = aval[2:n]/h/h
        c = aval[2:end-1]/h/h
        a = -(aval[1:n]+aval[2:end])/h/h
        rhs = constant(ones(n))
        us[i] = -solve_tridiagonal(a, b, c, rhs)
        us[i].set_shape((n,))
    end
    hcat(us...)'
end

function PDESample(mb_size)
    us = zeros(n,mb_size)
    μ, σ = get_sample(mb_size)
    for i = 1:mb_size
        ax = collect((1:2:2n+1)/2*1/(n+1))
        aval = avals.(ax, μ[i], σ[i])

        h = 1/(n+1)
        b = aval[2:n]/h/h
        c = aval[2:end-1]/h/h
        a = -(aval[1:n]+aval[2:end])/h/h
        rhs = ones(n)
        A = diagm(0=>a, -1=>b, 1=>c)
        us[:,i] = -A\rhs
    end
    us'|>Array
end 