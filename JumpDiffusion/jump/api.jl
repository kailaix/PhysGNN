using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
matplotlib.use("agg")

multivariate_jump_diffusion_ = load_op_and_grad("$(@__DIR__)/build/libJumpDiffusion","multivariate_jump_diffusion")
jump_diffusion_ = load_op_and_grad("$(@__DIR__)/build/libJumpDiffusion","jump_diffusion")

"""
    jd(y,tau,t,n,m,c)

- `y` : jump distribution; size = `N`
- `tau` : jump time; size = `N`
- `t` : terminal time; a scalar
- `n` : #grids-1 in the discretization; an integer scalar
- `mm` : #effective jump in `tau`; an integer scalar 
- `c` : jump magnitude; size = `n+N+1`
- `a, b, bp` : diffusion parts
- `dw` : random normal variables 

{tau, LinRange(0, t, n+1)} will be first sorted. 
"""
function jd(y,tau,t,n,mm,a,b,bp,c,dw)
end

"""
    mjd(y,tau,t,n,m,c)

- `y` : jump distribution; size = `2xN`
- `tau` : jump time; size = `N`
- `t` : terminal time; a scalar
- `n` : #grids-1 in the discretization; an integer scalar
- `m` : #effective jump in `tau`; an integer scalar 
- `c` : jump magnitude; size = `n+N+1`

{tau, LinRange(0, t, n+1)} will be first sorted. 
"""
function mjd(y,tau,t,n,m,c)
    y,tau,t,n,m,c = convert_to_tensor([y,tau,t,n,m,c], [Float64, Float64, Float64, Int32, Int32, Float64])
    multivariate_jump_diffusion_(y, tau, t, n, m, c)
end


function mjds(y,tau,t,n,m,c)
    y,tau,t,n,m,c = convert_to_tensor([y,tau,t,n,m,c], [Float64, Float64, Float64, Int32, Int32, Float64])
    S = map(o->multivariate_jump_diffusion_(o...), [y,tau,t,n,m,c], dtype = Float64)
end