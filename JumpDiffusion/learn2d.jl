include("jump/api.jl")
include("CommonFuncs.jl")
using PyPlot
matplotlib.use("agg")
using Statistics
using LinearAlgebra
using DelimitedFiles
using RollingFunctions
using MAT

# model = MixtureModel(
#     [MvNormal([0.0;1.0],0.2*I),
#     MvNormal([1.0;0.0],0.2*I),
#     MvNormal([-1.0;0.0],0.2*I)]
# )
# model = MixtureModel(
#     [MvNormal([0.0;1.0],0.2*I),
#     MvNormal([1.0;0.0],0.2*I)]
# )
reset_default_graph()
# cfun = t->1+t^2
mid = parse(Int64, ARGS[1])
ranges = [
    [(0.0,1.3),(0.0,1.8)],
    [(-3.0,3.0),(-2.0,4.0)],
    [(-0.5,3.5),(0.0,4.0)]
]
vranges = [
    (0.0,3.7),
    (0.0,0.45),
    (0.0,0.8)
]
sranges = [
    [(-8.0,1.0),(0.0,10.0)],
    [(-30.0,0.0), (-9.0,8.0)],
    [(0.0,14.0), (0.0,25.0)]
]
svranges = [
    (0.0,0.18),
    (0.0,0.03),
    (0.0,0.035)
]



latent_dim = 20
hidden_size = 20
batch_size = 64
n_layer = 4
cfun = t->1.0

# model = MvNormal([-0.5;0.5], 0.2*[1.0 1.0;1.0 2.0])


T = 1.0
max_size = Int(T)*30
N = 100

# generating `y` from neural networks
pt = constant(ones(batch_size)*T)
pn = constant(ones(Int32, batch_size)*N)
pms = placeholder(Int32, shape=[batch_size])
ptau = placeholder(Float64, shape=[batch_size, max_size])
pc = placeholder(Float64, shape=[batch_size, max_size+N+1])
z = placeholder(Float64, shape=[batch_size*max_size, latent_dim])
z_ = ae(z, [ones(Int64, n_layer)*hidden_size;2])
ys = tf.stack([reshape(z_[:,1],batch_size, max_size), reshape(z_[:,2],batch_size, max_size)], axis=1)

Sall = mjds(ys,ptau,pt,pn,pms,pc)
Sall.set_shape((batch_size, 2, N+1))
Sall_ = tf.gather(Sall, indices=[N], axis=2)
# Sall_ = tf.reshape(Sall, (batch_size,-1))
pV = placeholder(Float64, shape=[batch_size, 2, N+1])
pV_ = tf.gather(pV, indices=[N], axis=2)
# pV_ = tf.reshape(pV, (batch_size,-1))

loss = empirical_sinkhorn(squeeze(Sall_), 
                squeeze(pV_), dist=(x,y)->dist(x, y, 2); method="lp") 
opt = AdamOptimizer(0.002, beta1=0.5).minimize(loss)
# opt = RMSPropOptimizer().minimize(loss)

wgts = get_weights()
clip_op = clip(wgts, -0.1,0.1)
# error()
sess = Session(); init(sess)
Strue = matread("2D$mid/S.mat")["S"]

function vis(i)
    global tloss_
    SS_ = []
    YS_ = []
    tl_ = []
    for i = 1:150
        tau, ms, t, n, c = make_data2(λ=0.1, T=T, N=N, Ns=batch_size, max_size=max_size, cfun=cfun)
        idx = rand(1:10000, batch_size)
        dic = Dict(
            ptau=>tau, 
            pms=>ms,
            pV=>Strue[idx,:,:],
            z=>randn(batch_size*max_size, latent_dim),
            pc=>c
        )
        a, b, tl = run(sess, [ys, Sall, loss], feed_dict=dic)
        push!(SS_, b)
        push!(YS_, a)
        push!(tl_, tl)
    end
    push!(tloss_, mean(tl_))
    SS_ = vcat(SS_...)
    YS_ = vcat(YS_...)
    YS = zeros(size(YS_,1)*size(YS_,3), 2)
    k = 1
    for i = 1:size(YS_,1)
        for j = 1:size(YS_,3)
            YS[k,:] = YS_[i, :, j]
            k += 1
        end
    end
    
    close("all")
    hist2D(SS_[:,1,end], SS_[:,2, end], bins=30, normed=true,
        range=sranges[mid], vmin=svranges[mid][1], vmax=svranges[mid][2], rasterized=true)
    colorbar()
    savefig("2D$mid/S$i.png")
    savefig("2D$mid/S$i.pdf")

    close("all")
    hist2D(YS[:,1], YS[:,2], bins=30, normed=true,
            range=ranges[mid], vmin=vranges[mid][1], vmax=vranges[mid][2], rasterized=true)
    colorbar()
    savefig("2D$mid/Y$i.png")
    savefig("2D$mid/Y$i.pdf")

    writedlm("2D$mid/loss.txt", loss_)
    writedlm("2D$mid/tloss.txt", tloss_)
    # d = Dict("S"=>SS_, "Y"=>YS_, "loss"=>loss_)
    # matwrite("2D$mid/data$i.mat", d)
end


sess = Session(); init(sess)
# error()
loss_ = Float64[]
tloss_ = Float64[]
for i = 1:1001
    global loss_
    if mod(i,100)==1
        vis(i)
    end
    tau, ms, t, n, c = make_data2(λ=0.1, T=T, N=N, Ns=batch_size, max_size=max_size, cfun=cfun)
    idx = rand(1:10000, batch_size)
    dic = Dict(
        ptau=>tau, 
        pms=>ms,
        pV=>Strue[idx,:,:],
        z=>randn(batch_size*max_size, latent_dim),
        pc=>c
    )
    _, _, l = run(sess, [opt,clip_op,loss], feed_dict=dic)
    push!(loss_, l)
    @show i, l
end




# if !isdir("Data"); mkdir("Data"); end 
# writedlm("Data/S.txt", S_)

# close("all")
# hist2D(S_[:,1,end], S_[:,2,end], normed=true, bins=30)
# savefig("test2.png")

# means = zeros(N+1, 2)
# stds = zeros(N+1,2,2)
# for i = 1:N+1
#     means[i,:] = mean(S_[:,:,i], dims=1)
#     stds[i,:,:] = cov(S_[:,:,i])
# end

# close("all")
# plot(means[:,1], means[:,2])
# savefig("test3.png")

# close("all")
# plot(stds[:,1,1])
# plot(stds[:,1,2])
# plot(stds[:,2,2])
# savefig("test4.png")