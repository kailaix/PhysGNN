include("CommonFuncs.jl")
include("Jump/api.jl")
using PyPlot
matplotlib.use("agg")

y = MixtureModel(Normal[
   Normal(-1.0, 1.0),
   Normal(10.0, 1.0)])

λ = 0.1
Ns = 10000
tau, ms, t, n, a, b, bp, c, dw = make_data(λ,1.0,100,Ns)
maxval = maximum(ms)
ys = constant(rand(y, Ns, maxval))
S = jdsim2(ys,tau,t,n,ms,a,b,bp,c,dw)

sess = Session(); init(sess)
S_ = run(sess, S)

close("all");hist(S_[:,end], density=true, bins=30); savefig("test3.png")
