include("CommonFuncs.jl")
using Random; Random.seed!(233)

hmat_idx = 1
tid = 1
latent_dim = 20

if length(ARGS)==2
    global hmat_idx = parse(Int64, ARGS[1])
    global tid = parse(Int64, ARGS[2])
    global latent_dim = parse(Int64, ARGS[3])
end

@info hmat_idx, tid 
reset_default_graph()
m = 4
n = 2
h = 0.1
batch_size = 64

z = placeholder(Float64, shape=[batch_size*m*n,latent_dim])
Eμ, H = ae_Hmat(z, [20,20,20,20,2])
pH = placeholder(Float64, shape=[batch_size, m*n, 3, 3])
H = tf.reshape(H, (batch_size, m*n, 3, 3))
u = get_disps(H, m, n, h)
uexact = get_disps(pH, m, n, h)

# gradients(sum(u), H)
loss = empirical_sinkhorn(u, uexact, dist=(x,y)->dist(x, y, 1), method="lp")

opt = AdamOptimizer(0.0002,beta1=0.5).minimize(loss)
# opt = AdamOptimizer(0.001,beta1=0.5).minimize(loss)

sess = Session(); init(sess)

Hs = zeros(batch_size,m*n,3,3)
for i = 1:batch_size
    Hs[i,:,:,:] = get_random_mat2(hmat_idx)
end
dic=Dict(z=>randn(batch_size*m*n,latent_dim),
                    pH=>Hs)
@info run(sess, loss, feed_dict=dic)


fixed_z = randn(100,batch_size*m*n,latent_dim)
fixed_Hs = zeros(100,batch_size,m*n,3,3)
for k = 1:100
    for i = 1:batch_size
        fixed_Hs[k,i,:,:,:] = get_random_mat2(hmat_idx)
    end
end

res1 = Result("nn$hmat_idx$tid")
plots = [1, 11, 51, 101]
for i = 1:10001
    Hs = zeros(batch_size,m*n,3,3)
    for i = 1:batch_size
        Hs[i,:,:,:] = get_random_mat2(hmat_idx)
    end
    dic=Dict(z=>randn(batch_size*m*n,latent_dim),
                        pH=>Hs)
    l, _ = run(sess, [loss, opt], feed_dict=dic)

    if i in plots || mod(i,500)==1
        tl = zeros(100)
        res = []
        for k = 1:100
            dic2=Dict(z=>fixed_z[k,:,:],
                                pH=>fixed_Hs[k,:,:,:,:])
            eμ, tl[k] = run(sess, [Eμ,loss], feed_dict=dic2)
            push!(res, eμ)
        end
        res = vcat(res...)
        visualize(res[:,1], res[:,2]); savefig("nn$hmat_idx$tid/res$i.pdf",rasterized=true)
        save_result(res1, i, l, mean(tl), std(tl))
        plot(res1)
    end
    @show i, l
end

@save "nn$hmat_idx$tid.jld2" res1