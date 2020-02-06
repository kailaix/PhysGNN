include("jump/api.jl")
include("CommonFuncs.jl")
using PyPlot
matplotlib.use("agg")
using Statistics
using LinearAlgebra
using DelimitedFiles
using MAT

# cfun = t->1+t^2
cfun = t->1.0

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

struct MyDist end
function Base.:rand(m::MyDist, n::Int64)
    Z = zeros( 2, n)
    i = 1
    while i<=n
        r = rand()*0.3 .+ 1.5
        θ = rand()*0.2*π .+ 0.25π
        Z[:,i] = r*[cos(θ) sin(θ)]
        i = i + 1
    end
    Z
end
model1 = MyDist()

# works
model2 = MvNormal([0.0;1.0], 0.2*[2.0 1.0;1.0 2.0])

# works
model3 = MvNormal([1.5;2.0], 0.2*[1.0 0.0;0.0 1.0])

models = [model1, model2, model3]
model = models[mid]


T = 1.0
max_size = Int(T)*30
N = 100
Ns = 10000
tau, ms, t, n, c = make_data2(λ=0.1, T=T, N=N, Ns=Ns, max_size=max_size, cfun=cfun)

y = zeros(Ns, 2, N)
for i = 1:Ns
    y[i,:,:] = rand(model, N)
end
S = mjds(y,tau,t,n,ms,c)
sess = Session(); init(sess)
S_ = run(sess, S)

rm("2D$mid", recursive=true, force=true); mkdir("2D$mid")
matwrite("2D$mid/S.mat", Dict("S"=>S_))


close("all")
ys_ = zeros(Ns*N, 2)
k = 0
for i = 1:Ns
    for j = 1:N 
        global k += 1
        ys_[k,:] = y[i,:,j]
    end
end
hist2D(ys_[:,1], ys_[:,2], normed=true, bins=30, range=ranges[mid], rasterized=true,
            vmin=vranges[mid][1], vmax=vranges[mid][2])
colorbar()

savefig("2D$mid/ys.png")

close("all")
hist2D(S_[:,1,end], S_[:,2,end], normed=true, bins=30,  rasterized=true,
        range=sranges[mid], vmin=svranges[mid][1], vmax=svranges[mid][2])
colorbar()

savefig("2D$mid/S.png")

