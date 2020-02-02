include("CommonFuncs.jl")
using Random; Random.seed!(233)

hmat_idx = 3
tid = 0

if length(ARGS)==2
    global hmat_idx = parse(Int64, ARGS[1])
    global tid = parse(Int64, ARGS[2])
end
@info hmat_idx, tid 
reset_default_graph()
m = 4
n = 2
h = 0.1
idx = rand(1:(m+1)*(n+1), 10)
idx = [idx; idx.+(m+1)*(n+1)]
A = Variable(zeros(2,2))
μ = Variable([1.5;0.25])
z = placeholder(Float64, shape=[16*m*n,2])
Eμ, H = gs_Hmat(z, A, μ)
pH = placeholder(Float64, shape=[16, m*n, 3, 3])
H = tf.reshape(H, (16, m*n, 3, 3))
u = get_disps(H, m, n, h)
uexact = get_disps(pH, m, n, h)

# gradients(sum(u), H)
loss = empirical_sinkhorn(u, uexact, dist=(x,y)->dist(x, y, 1), method="lp")

opt = AdamOptimizer(0.0002,beta1=0.5).minimize(loss)
sess = Session(); init(sess)

Hs = zeros(16,m*n,3,3)
for i = 1:16
    Hs[i,:,:,:] = get_random_mat2(hmat_idx)
end
dic=Dict(z=>randn(16*m*n,2),
                    pH=>Hs)
@info run(sess, loss, feed_dict=dic)


fixed_z = randn(100,16*m*n,2)
fixed_Hs = zeros(100,16,m*n,3,3)
for k = 1:100
    for i = 1:16
        fixed_Hs[k,i,:,:,:] = get_random_mat2(hmat_idx)
    end
end

res1 = Result("gs$hmat_idx$tid")
plots = [1, 10, 50, 100, 200, 500]
for i = 1:15001
    Hs = zeros(16,m*n,3,3)
    for i = 1:16
        Hs[i,:,:,:] = get_random_mat2(hmat_idx)
    end
    dic=Dict(z=>randn(16*m*n,2),
                        pH=>Hs)
    l, _ = run(sess, [loss, opt], feed_dict=dic)

    if i in plots || mod(i,1000)==1
        tl = zeros(100)
        res = []
        for k = 1:100
            dic2=Dict(z=>fixed_z[k,:,:],
                                pH=>fixed_Hs[k,:,:,:,:])
            eμ, tl[k] = run(sess, [Eμ,loss], feed_dict=dic2)
            push!(res, eμ)
        end
        res = vcat(res...)
        visualize(res[:,1], res[:,2]); savefig("gs$hmat_idx$tid/res$i.pdf", rasterized=true)
        save_result(res1, i, l, mean(tl), std(tl))
        plot(res1)
    end
    @show i, l
end

@save "nn$hmat_idx$tid.jld2" res1