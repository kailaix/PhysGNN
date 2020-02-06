include("CommonFuncs.jl")

reset_default_graph()

mb_size = 32
X_dim = 100
z_dim = 10
h_dim = 128

# function sample_z(m, n)
#     return randn(m, n)
# end

function sample_z(m, n)
    return (rand(m,n) .- 0.5)*2
end

function generator(z)
    ae(z, [20,20,20,2], "generator")
end

z = placeholder(Float64, shape=[nothing, z_dim])
G_out = abs(generator(z)) # 32 x 2
G_sample = PDE(G_out) # 32 x 100

true_sample = placeholder(Float64, shape=(32, n))

loss = empirical_sinkhorn(G_sample, true_sample, dist=(x,y) -> dist(x, y, 2), method="lp")
opt = AdamOptimizer(0.001, beta1=0.5).minimize(loss)

sess = Session()
init(sess)

if !isdir("nnmclogs$(dist_id)")
    mkdir("nnmclogs$(dist_id)")
end
_loss = Float64[]
_tloss = Float64[]
for it=1:1000000
    # if mod(it, 100)==1
    #     samples = run(sess, G_out, feed_dict=Dict( z=>sample_z(10000, z_dim)))
    #     writedlm("nnmclogs$(dist_id)/$it.txt", samples)
    #     close("all")
    #     hist2D(samples[:,1],abs.(samples[:,2]), normed=true, bins=50, range=((0.0,1.0),(0.0,1.0)), vmin=0.0, vmax=5.0)
    #     colorbar()
    #     savefig("nnmclogs$(dist_id)/$(it)_sample.png")
    #     if it>1
    #         close("all")
    #         plot(_loss, label="Generator")
    #         yscale("log")
    #         savefig("nnmclogs$(dist_id)/loss.png")
    #         xlabel("Iterations")
    #         ylabel("Loss")
    #         writedlm("loss.txt", _loss)
    #     end
    # end

    
    # tloss = run(sess, loss,  feed_dict=Dict(true_sample=>PDESample(32), z=>sample_z(32, z_dim)))
    if length(_loss)==0 || mod(it, 200)==0
        samples = run(sess, G_out, feed_dict=Dict( z=>sample_z(10000, z_dim)))
        writedlm("nnmclogs$(dist_id)/$it.txt", samples)
        close("all")
        hist2D(samples[:,1],abs.(samples[:,2]), normed=true, bins=50, range=((0.0,1.0),(0.0,1.0)), vmin=0.0, vmax=vmax)
        colorbar()
        savefig("nnmclogs$(dist_id)/$(it)_sample.png")
        
        L = 0
        for j = 1:10
            l_ = run(sess, loss,
                feed_dict=Dict(true_sample=>PDESample(mb_size), z=>sample_z(mb_size, z_dim)))
            L += l_ 
        end
        global tloss = L/10
        push!(_tloss, tloss)
        
        if it>1
            close("all")
            plot(_loss, label="Train")
            plot(200*(0:length(_tloss)-1), _tloss, "-o", label="Validation")
            legend()
            yscale("log")
            xlabel("Iterations")
            ylabel("Loss")
            savefig("nnmclogs$(dist_id)/loss.png")
            savefig("nnmclogs$(dist_id)/loss.pdf")
            writedlm("nnmclogs$(dist_id)/loss.txt", _loss)
            writedlm("nnmclogs$(dist_id)/tloss.txt", _tloss)
        end
    end
    X_mb = PDESample(mb_size)
    l,_ = run(sess, [loss, opt], 
        feed_dict=Dict(true_sample=>X_mb, z=>sample_z(mb_size, z_dim)))
    push!(_loss, l)
    # push!(_tloss, l)
    @show it, l, tloss
end
