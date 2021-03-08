using AdFem
using PyPlot 
using PyCall

function simulation(μ)

    mmesh = Mesh(10,10,0.1)
    κ = (1+μ) * ones(get_ngauss(mmesh))
    A = compute_fem_laplace_matrix1(κ, mmesh)
    F = eval_f_on_gauss_pts((x, y)->2π^2 * sin(π*x) * sin(π*y), mmesh)
    rhs = compute_fem_source_term1(F, mmesh)
    bdnode = bcnode(mmesh)
    A, rhs = impose_Dirichlet_boundary_conditions(A, rhs, bdnode, zeros(length(bdnode)))
    sol = A\rhs 

    # sol[97]
end

mutable struct Environment 
    μ::PyObject
    dnn::PyObject
    obs::PyObject
    sess::PyObject
end

function Environment(sess, batch_size=16)
    μ = placeholder(Float64, shape = [batch_size])
    OBS = []
    for i = 1:batch_size
        sol = simulation(μ[i])
        obs = sol[97]
        push!(OBS, obs)
    end
    obs = vcat(OBS...)
    Environment(μ, PyNULL(), obs, sess)
end

function Simulator(sess, batch_size=16)
    μ = placeholder(Float64, shape = [batch_size,10])
    dnn = squeeze(fc(μ, [20,20,20,1], "simulator"))+2
    OBS = []
    for i = 1:batch_size
        sol = simulation(dnn[i])
        obs = sol[97]
        push!(OBS, obs)
    end
    obs = vcat(OBS...)
    Environment(μ, dnn, obs, sess)
end

function sample_exact(e::Environment)
    n = length(e.μ)
    w = abs.(0.3*randn(n) .+ 1.0)
end

function sample_latent(e::Environment)
    randn(size(e.μ)...)
end

function ADCME.:sample(e::Environment)
    w = sample_exact(e)
    val = run(e.sess, e.obs, e.μ=>w)
end

function (e::Environment)(w::Array{T,1}) where T<:Real
    @assert length(w)==length(e.μ)
    val = run(e.sess, e.obs, e.μ=>w)
end

function ResultSet(filename::String = "result.db"; overwrite = false)
    db = Database(filename)
    if overwrite
        execute(db, """DROP TABLE IF EXISTS result""")
    end
    execute(db, """
CREATE TABLE result (
    iter integer, 
    dnn real,
    loss real 
)
""")
    db
end

function save_to_db(db::Database, i::Int64, loss::Float64, dnn::Array{Float64,1}) 
    data = [(s,) for s in dnn]
    execute(db, """
INSERT INTO result VALUES ($i,?,$loss)
    """, data)
end

ss = pyimport("sklearn.neighbors")
function fit_kde(data, bandwidth=0.2)
    data = reshape(data, :, 1)
    kde = ss.KernelDensity(kernel = "gaussian", bandwidth = bandwidth).fit(data)
    kde.score_samples
end