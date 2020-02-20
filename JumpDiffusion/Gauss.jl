"""
This script demonstrates how to learn a mixture Gaussian distribution using LP 
"""

include("CommonFuncs.jl")
using PyPlot
using PyCall
sg = pyimport("sklearn.datasets.samples_generator")
sd = pyimport("sklearn.datasets")
matplotlib.use("agg")
using LinearAlgebra
using MAT


# latent_dim = 100
# hidden_size = 20
# batch_size = 64
# n_layer = 4
# model_id = 4

latent_dim = 100
hidden_size = parse(Int64, ARGS[1])
batch_size = 64
n_layer = parse(Int64, ARGS[2])
model_id = parse(Int64, ARGS[3])

struct MyDist end
function Base.:rand(m::MyDist, n::Int64)
    sg.make_moons(n, noise=0.1)[1]'
end

struct MyDist2 end
function Base.:rand(m::MyDist2, n::Int64)
    sg.make_s_curve(n, noise=0.2)[1][:,[1;3]]'
end

struct MyDist3 end
function Base.:rand(m::MyDist3, n::Int64)
    sd.make_swiss_roll(n, noise=1.0)[1][:,[1;3]]'
end



model1 = MyDist()
model2 = MyDist2()
model3 = MyDist3()
model4 = MixtureModel(
    [MvNormal([0.0;1.5],0.5*I),
    MvNormal([1.5;0.0],0.5*I),
    MvNormal([-1.5;0.0],0.5*I)]
)
models = [model1, model2, model3, model4]

m = models[model_id]

ranges = [
    [(-1.5,2.5), (-0.75,1.25)],
    [(-1.5,1.5), (-3,3)],
    [(-15,15), (-15,15)],
    [(-3,3), (-2,3)]
]
vranges = [
    (0.0,0.8),
    (0.0,0.28),
    (0.0,0.009),
    (0.0,0.225)
]
# rm("figure$(model_id)_$(n_layer)_$(hidden_size)", recursive=true, force=true)
# mkdir("figure$(model_id)_$(n_layer)_$(hidden_size)")
# mkdir("figure$(model_id)_$(n_layer)_$(hidden_size)/pdf")
reset_default_graph()
batch_size = 64

z = placeholder(Float64, shape=[batch_size, latent_dim])
z_ = ae(z, [ones(Int64, n_layer)*hidden_size;2])
w_ = placeholder(Float64, shape=[batch_size, 2])



# m = MyDist3()
!isdir("figure$(model_id)_$(n_layer)_$(hidden_size)") && mkdir("figure$(model_id)_$(n_layer)_$(hidden_size)")
!isdir("figure$(model_id)_$(n_layer)_$(hidden_size)/pdf") && mkdir("figure$(model_id)_$(n_layer)_$(hidden_size)/pdf")

M = rand(m, 50000)
close("all")
hist2D(M[1,:],M[2,:],normed=true,bins=50, vmin=vranges[model_id][1], 
    vmax=vranges[model_id][2], range=ranges[model_id],rasterized=true)
axis("off")
axis("equal")
# colorbar()
savefig("figure$(model_id)_$(n_layer)_$(hidden_size)/gauss.png")
savefig("figure$(model_id)_$(n_layer)_$(hidden_size)/pdf/gauss.pdf")

error()
loss = empirical_sinkhorn(z_, w_, dist=(x,y)->dist(x,y,2); method="lp")
opt = AdamOptimizer(0.001, beta1=0.5).minimize(loss)

function vis(i)
    Vs = []
    for i = 1:1000
        dic = Dict(
            z => randn(batch_size, latent_dim),
        )
        zz = run(sess, z_, feed_dict=dic)
        push!(Vs, zz)
    end
    Vs = vcat(Vs...)
    close("all")
    hist2D(Vs[:,1], Vs[:,2], normed=true, bins=30, 
        vmin=vranges[model_id][1], vmax=vranges[model_id][2], range=ranges[model_id], rasterized=true)
    axis("off")
    axis("equal")
    # colorbar();
    savefig("figure$(model_id)_$(n_layer)_$(hidden_size)/gauss$i.png")
    savefig("figure$(model_id)_$(n_layer)_$(hidden_size)/pdf/gauss$i.pdf")
end

sess = Session(); init(sess)
_loss = Float64[]
for i = 1:15001
    if mod(i,3000)==1
        vis(i)
    end
    dic = Dict(
        z => randn(batch_size, latent_dim),
        w_=> Array(rand(m, batch_size)')
    )
    _, l = run(sess, [opt, loss], feed_dict=dic)
    push!(_loss, l)
    @show i, l 
end
matwrite("figure$(model_id)_$(n_layer)_$(hidden_size)/loss.mat", Dict("loss"=>_loss))