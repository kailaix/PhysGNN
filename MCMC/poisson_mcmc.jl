using AdFem
using PyPlot 
include("common.jl")


db = Database("result.db")
conn = execute(db, """
SELECT * FROM exact
""")
result = collect(conn)
mu = [x[1] for x in result]
obs = [x[2] for x in result]
close(db)


kde = fit_kde(obs, 0.01)
x0 = reshape(collect(LinRange(0, 1.0, 100)), :, 1)
exc = exp.(kde(x0))
figure()
hist(obs, bins=30, facecolor="b", edgecolor="k", density=true)
ylabel("Density", fontsize=14)
xlabel(L"$u$(0.8, 0.8)", fontsize=14)
# plot(x0, exc, label = "Data")
savefig("hist_poisson_data.pdf")
savefig("hsit_poisson_data.png")

mmesh = Mesh(10,10,0.1)

sess = Session(); init(sess)

num_samples = 20000
burnin_ratio = 0.2

func_sample = Environment(sess, 1)
function prob(x::Float64)
    # μ0 = 0.1745
    # σ0 = 0.0281
    # return 1/(σ0*2*π) * exp(-1/2*((x-μ0)/σ0)^2)
    global kde
    return exp.(kde(reshape([x],1,1)))[1]
end
function next_sample(x::Float64)
    return randn(1)[1]*0.03 + x
end

μ_prev = 1.0
μ_samples = []
y_samples = []

# func_sample = Environment(sess, 100)
# # y = func_sample(ones(100))
# y = sample_exact(func_sample)
# figure()
# hist(y, bins=30)
# savefig("debug.png")


for i in 1:num_samples
    global μ_prev
    μ = next_sample(μ_prev)

    accept_ratio = prob(func_sample([μ])[1]) / prob(func_sample([μ_prev])[1])
    if rand() <= accept_ratio
        μ_prev = μ
    end

    if i > num_samples * burnin_ratio
        push!(μ_samples, μ_prev)
        push!(y_samples, func_sample([μ_prev])[1])
    end
end

db = Database("result.db")
execute(db, "drop table mcmc")
execute(db, """
CREATE TABLE mcmc (
    mu real, 
    obs real 
)
""")
execute(db, """
INSERT INTO mcmc VALUES (?,?)
""", [(μ_samples[i], y_samples[i]) for i = 1:length(μ_samples)])
close(db)

figure()
hist(μ_samples, bins=30)
savefig("mcmc_u.png")

figure()
hist(y_samples)
savefig("mcmc_y.png")


kde = fit_kde(mu, 0.1)
x0 = reshape(collect(LinRange(0, 2, 100)), :, 1)
exc = exp.(kde(x0))
figure()
plot(x0, exc, label = L"True \mu")

kde_ = fit_kde(μ_samples, 0.1)
exc_ = exp.(kde_(x0))
# hist(obs)
plot(x0, exc_, label = L"Estimated \mu")
legend()
savefig("kde_mu_mcmc.png")

# env(randn(10))

# env = 
# z = randn(10)
# dnn = fc(z, [20,20,20,1])|>squeeze 
# κ = 1+dnn
# A = compute_fem_laplace_matrix1(κ, mmesh)
# F = eval_f_on_gauss_pts((x, y)->2π^2 * sin(π*x) * sin(π*y), mmesh)
# rhs = compute_fem_source_term1(F, mmesh)
# bdnode = bcnode(mmesh)
# A, rhs = impose_Dirichlet_boundary_conditions(A, rhs, bdnode, zeros(length(bdnode)))
# sol = A\rhs 



# SOL = run(sess, sol)
# close("all")
# visualize_scalar_on_fem_points(SOL, mmesh)
# savefig("poisson-dnn.png")

