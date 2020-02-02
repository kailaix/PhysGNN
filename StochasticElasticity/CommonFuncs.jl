using Revise
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using MAT
using PyCall
using JLD2

mutable struct Result 
    dname::String 
    ii::Array{Int64}
    loss::Array{Float64}
    tloss::Array{Float64}
    tloss_std::Array{Float64}
end

function Result(dname::String)
    rm(dname, recursive=true, force=true)
    mkdir(dname)
    Result(dname, [], [], [], [])
end


function save_result(res::Result, i::Int64, l::Float64, tl::Float64, v::Float64)
    push!(res.ii, i)
    push!(res.tloss, l)
    push!(res.tloss_std, v)
end

function PyPlot.:plot(res::Result)
    if length(res.loss)==0
        return
    end
    close("all")
    vs = res.ii 
    plot(vs, res.loss[vs], "g", label="Training")
    plot(vs, res.tloss, "r", label="Testing")
    fill_between(vs, res.tloss-res.tloss_std, res.tloss+res.tloss_std, alpha=0.5, color="orange")
    xlabel("Iterations")
    ylabel("Loss")
    savefig(joinpath(res.dname, "loss.png"))
end


function get_H_mat(E, ν)
    [
        E/(1-ν^2) ν*E/(1-ν^2) 0.0
        ν*E/(1-ν^2) E/(1-ν^2) 0.0
        0.0 0.0 E/(2(1+ν))
    ]
end

function plane_stress_hmat(e,nu)
    plane_stress_hmat_ = load_op_and_grad("$(@__DIR__)/PlaneStressHmat/build/libPlaneStressHmat","plane_stress_hmat")
    e,nu = convert_to_tensor([e,nu], [Float64,Float64])
    plane_stress_hmat_(e,nu)
end

function compute_fem_traction_term(t::Array{Float64, 2},
    bdedge::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    @assert size(t,1)==size(bdedge,1) || size(t,2)==2
    rhs = zeros(2*(m+1)*(n+1))
    for k = 1:size(bdedge, 1)
        ii, jj = bdedge[k,:]
        rhs[ii] += t[k,1]*0.5*h 
        rhs[jj] += t[k,1]*0.5*h
        rhs[ii+(m+1)*(n+1)] += t[k,2]*0.5*h
        rhs[jj+(m+1)*(n+1)] += t[k,2]*0.5*h 
    end
    rhs
end

function spatial_fem_stiffness(H,m,n,h)
    spatial_fem_stiffness_ = load_op_and_grad("$(@__DIR__)/SpatialFemStiffness/build/libSpatialFemStiffness",
                        "spatial_fem_stiffness", multiple=true)
    H,m,n,h = convert_to_tensor([H,m,n,h], [Float64,Int32,Int32,Float64])
    ii,jj,vv = spatial_fem_stiffness_(H,m,n,h)
    ii,jj,vv
end

function fem_impose_Dirichlet_boundary_condition(ii,jj,vv,bd,m,n,h)
    dirichlet_bd_ = load_op_and_grad("$(@__DIR__)/DirichletBD/build/libDirichletBd","dirichlet_bd", multiple=true)
    ii,jj,vv,bd,m,n,h = convert_to_tensor([ii,jj,vv,bd,m,n,h], [Int64,Int64,Float64,Int32,Int32,Int32,Float64])
    ii1,jj1,vv1,_,_,_ = dirichlet_bd_(ii,jj,vv,bd,m,n,h)
    ii1,jj1,vv1
end

# py"""
# import traceback
# import tensorflow as tf
# try:
#     tf.gradients(tf.reduce_sum($vv1), $H) 
# except Exception:
#     print(traceback.format_exc())
# """
function get_disp(H, m, n, h; tscale = 0.3)
    bd = Int64[]
    for j = 1:n+1 
        push!(bd,(j-1)*(m+1)+1)
    end
    bdedge = zeros(Int64, n,2)
    for j = 1:n 
        bdedge[j,:] = [(j-1)*(m+1)+m+1; j*(m+1)+m+1]
    end
    t = zeros(n,2)
    for i = 1:n 
        t[i,1] = tscale
    end
    ii,jj,vv = spatial_fem_stiffness(H, m, n, h)
    ii1,jj1,vv1 = fem_impose_Dirichlet_boundary_condition(ii,jj,vv, bd, m, n, h)
    rhs = compute_fem_traction_term(t, bdedge, m, n, h)
    rhs[[bd;bd.+(m+1)*(n+1)]] .= 0.0
    A = SparseTensor(ii1,jj1,vv1,2(m+1)*(n+1),2(m+1)*(n+1))
    u = A\rhs
end

function get_disps(H, m, n, h; tscale = 0.3)
    H = convert_to_tensor(H, dtype=Float64)
    u = map(H->get_disp(H, m, n, h; tscale = tscale), H)
end

function PyPlot.:scatter(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64)
    U = zeros(n+1, m+1)
    V = zeros(n+1, m+1)
    X = zeros(n+1, m+1)
    Y = zeros(n+1, m+1)
    for i = 1:m+1
        for j = 1:n+1
            U[j,i] = u[(j-1)*(m+1)+i] 
            V[j,i] = u[(j-1)*(m+1)+i+(m+1)*(n+1)]
            X[j,i] = h*(i-1)
            Y[j,i] = h*(j-1)
        end
    end
    scatter((X+U)[:], (Y+V)[:])
end

function PyPlot.:scatter(u::Array{Float64,2}, m::Int64, n::Int64, h::Float64)
    for i = 1:size(u,1)
        scatter(u[i,:], m, n, h)
    end
end

function get_random_mat()
    E = 1.0
    ν = 0.35
    m = 10
    n = 5
    h = 0.1
    H = zeros(m*n, 3, 3)
    for i = 1:m*n
        O = rand(3,3)
        O = O'*O 
        H[i,:,:] = get_H_mat(E, ν) + O 
    end
    H 
end

function get_random_mat2(idx=1)
    local u, v
    E = 1.0
    ν = 0.25
    H = zeros(m*n, 3, 3)
    for i = 1:m*n
        # 
        if idx==1
            if rand()>0.5
                u = (rand()-0.5)*0.25 + 0.25
                v = (rand()-0.5)*0.25 + 0.25 
            else
                u = (rand()-0.5)*0.25 + 0.75
                v = (rand()-0.5)*0.25 + 0.75
            end
            H[i,:,:] = get_H_mat(u+ 1.0, v*0.5)
        elseif idx==2
            u = 0.2*randn()
            H[i,:,:] = get_H_mat(u+ 1.5, 0.4)
        elseif idx==3
            u = rand() + 1.0
            v = (u-1.0)*0.35 + 0.1
            H[i,:,:] = get_H_mat(u, v) 
        end
    end
    H 
end

function ae_Hmat(z, config::Array{Int64} = [20,20,20,2], name::String="default")
    z = convert_to_tensor(z, dtype=Float64)
    out = ae(z, config, name)
    out = [out[:,1]+1.5 sigmoid(out[:,2])*0.5]
    out, plane_stress_hmat(out[:,1], out[:,2])
end

function gs_Hmat(z, A::PyObject, μ::PyObject)
    out = z*A .+ μ
    out, plane_stress_hmat(out[:,1], out[:,2])
end

function visualize(E, μ)
    close("all")
    hist2D(E, μ, bins=50, range=((1.0,2.0), (0.0,0.5)), density=true)
    colorbar()
    xlabel("E")
    ylabel("\$\\nu\$")
end

