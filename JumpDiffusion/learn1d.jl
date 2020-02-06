include("CommonFuncs.jl")
using PyPlot
matplotlib.use("agg")



# Generating True Data
tid = 1

if length(ARGS)>0
    global tid = parse(Int64, ARGS[1])
end
models = [ MixtureModel(Normal[
   Normal(-1.0, 0.2),
   Normal(5.0, 0.2)]),
   MixtureModel(Normal[
    Normal(-1.0, 0.2),
    Normal(1.0, 0.2),
    Normal(5.0, 0.2)]),
    MixtureModel(Normal[
    Normal(-1.0, 0.2),
    Normal(5.0, 0.2)], [0.9; 0.1])]

y = models[tid]

λ = 0.1
Ns = 10000
tau, ms, t, n, a, b, bp, c, dw = make_data(λ,1.0,100,Ns)
maxval = maximum(ms)
ys = constant(rand(y, Ns, maxval))
S = jdsim2(ys,tau,t,n,ms,a,b,bp,c,dw)
# V = S[:,end]
sess = Session(); init(sess)
V_ = run(sess, S)


# Learning the jump diffusion 
max_size = 30
batch_size = 64
pa = constant(zeros(batch_size, max_size+101))
pb = constant(zeros(batch_size, max_size+101))
pbp = constant(zeros(batch_size, max_size+101))
pc = constant(ones(batch_size, max_size+101))

pt = ones(batch_size)
pn = ones(Int32, batch_size)*100

pdw = placeholder(Float64, shape=[batch_size, max_size+101])
pms = placeholder(Int32, shape=[batch_size])
ptau = placeholder(Float64, shape=[batch_size, max_size])
z = placeholder(Float64, shape=[batch_size*30, 20])

ys = reshape(ae(z, [20,20,20,20,1]), batch_size, 30) 
Sall = jdsim2(ys,ptau,pt,pn,pms,pa,pb,pbp,pc,pdw)
# Send = reshape(Sall[:, end], -1, 1)
pV = placeholder(Float64, shape=[batch_size, 101])
loss = empirical_sinkhorn(Sall, pV, dist=(x,y)->dist(x, y, 2); method="lp") 

opt = AdamOptimizer(0.002, beta1=0.5).minimize(loss)
Send = Sall[:,end]


if !isdir("figures$tid/")
    mkdir("figures$tid/")
end
function vis(iter)
    # output matching
    Vs = []
    
    for i = 1:100
        @info i 
        tau, ms, t, n, a, b, bp, c, dw = make_data(λ, 1.0, 100, batch_size; max_size=max_size)
        idx = rand(1:10000, batch_size)
        dic = Dict(
            ptau=>tau, 
            pms=>ms,
            pdw=>dw, 
            z=>randn(batch_size*30, 20),
            pV=>V_[idx,:]
        )
        s_ = run(sess, Send, feed_dict=dic)
        push!(Vs, s_[:])
    end
    Vs = vcat(Vs...)
    close("all")
    hist(Vs, bins=30, density = true, alpha=0.5)
    hist(V_[:,end], bins=30, density = true, alpha=0.5)
    savefig("figures$tid/Send$iter.png")



    # input matching
    Vs = []
    for i = 1:100
        tau, ms, t, n, a, b, bp, c, dw = make_data(λ, 1.0, 100, batch_size; max_size=max_size)
        idx = rand(1:10000, batch_size)
        dic = Dict(
            ptau=>tau, 
            pms=>ms,
            pdw=>dw, 
            z=>randn(batch_size*30, 20),
            pV=>V_[idx,:]
        )
        s_ = run(sess, ys, feed_dict=dic)
        push!(Vs, s_[:])
    end
    Vs = vcat(Vs...)
    close("all")
    hist(Vs, bins=30, density = true, alpha=0.5)
    hist(rand(y,5000), bins=30, density = true, alpha=0.5)
    savefig("figures$tid/ys$iter.png")
end


sess = Session(); init(sess)

for i = 1:1000000
    if mod(i,1000)==1
        vis(i)
    end
    tau, ms, t, n, a, b, bp, c, dw = make_data(λ, 1.0, 100, batch_size; max_size=max_size)
    idx = rand(1:10000, batch_size)
    dic = Dict(
        ptau=>tau, 
        pms=>ms,
        pdw=>dw, 
        z=>randn(batch_size*30, 20),
        pV=>V_[idx,:]
    )
    _, l = run(sess, [opt, loss], feed_dict=dic)
    @show i, l
    # @show pv, se
end
