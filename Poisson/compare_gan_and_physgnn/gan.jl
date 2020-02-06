include("CommonFuncs.jl")

reset_default_graph()
mb_size = 32
X_dim = 100
z_dim = 10
h_dim = 128

x = placeholder(Float64, shape = [nothing, X_dim])
# D_W1 = Variable(xavier_init([X_dim, h_dim]))
# D_b1 = Variable(zeros(h_dim))
# D_W2 = Variable(xavier_init([h_dim, 1]))
# D_b2 = Variable(zeros(1))
# theta_D = [D_W1, D_b1, D_W2, D_b2]

z = placeholder(Float64, shape=[nothing, z_dim])
# G_W1 = Variable(xavier_init([z_dim, h_dim]))
# G_b1 = Variable(zeros(h_dim))
# G_W12 = Variable(xavier_init([h_dim, h_dim]))
# G_b12 = Variable(zeros(h_dim))
# G_W13 = Variable(xavier_init([h_dim, h_dim]))
# G_b13 = Variable(zeros(h_dim))
# G_W2 = Variable(xavier_init([h_dim, 2]))
# G_b2 = Variable(zeros(2))
# theta_G = [G_W1, G_W2, G_b1, G_b2]

function sample_z(m, n)
    return (rand(m,n) .- 0.5)*2
end

# function generator(z)
#     G_h1 = relu(z*G_W1+G_b1)
#     G_h1 = relu(G_h1*G_W12+G_b12)
#     G_h1 = relu(G_h1*G_W13+G_b13)
#     G_log_prob = G_h1*G_W2 + G_b2
#     G_log_prob
# end

# function discriminator(x)
#     D_h1 = relu(x*D_W1+D_b1)
#     out = D_h1*D_W2 + D_b2
# end

function generator(z)
    ae(z, [20,20,20,2], "generator")
end

function discriminator(x)
    ae(x, [20,20,20,1], "discriminator")
end

G_out = abs(generator(z))
G_sample = PDE(G_out)
D_real = discriminator(x)
D_fake = discriminator(G_sample)

theta_G = [x for x in get_collection() if occursin("generator", x.name)]
theta_D = [x for x in get_collection() if occursin("discriminator", x.name)]


D_loss = mean(D_real) - mean(D_fake)
G_loss = -mean(D_fake)

D_solver = minimize(RMSPropOptimizer(1e-4), -D_loss, var_list=theta_D)
G_solver = minimize(RMSPropOptimizer(1e-4), G_loss, var_list=theta_G)
# σ_solver = minimize(AdamOptimizer(), G_loss, var_list=[σ])

clip_D = assign(theta_D, [clip(p, -0.01,0.01) for p in theta_D])
sess = Session()
init(sess)

if !isdir("analogs$(dist_id)")
    mkdir("analogs$(dist_id)")
end
_gloss = Float64[]
_dloss = Float64[]
for it=1:1000000
    if mod(it, 500)==1
        samples = run(sess, G_out, feed_dict=Dict( z=>sample_z(10000, z_dim)))
        writedlm("analogs$(dist_id)/$it.txt", samples)
        close("all")
        hist2D( samples[:,1],abs.(samples[:,2]), normed=true, 
                    bins=50, range=((0.0,1.0),(0.0,1.0)), vmin=0.0, vmax=vmax)
        colorbar()
        savefig("analogs$(dist_id)/$(it)_sample.png")
        close("all")
        plot(_gloss, label="Generator")
        plot(_dloss, label="Discriminator")
        legend()

        xlabel("Iterations")
        ylabel("Loss")
        savefig("analogs$(dist_id)/loss.png")
        writedlm("analogs$(dist_id)/loss.txt", [_gloss _dloss])
    end

    for k = 1:5
        # X_mb = randn(mb_size, X_dim) .+ 1.0  # generate sample
        X_mb = PDESample(mb_size)
        global _, Dl,_ = run(sess, [D_solver, D_loss, clip_D], 
            feed_dict=Dict(x=>X_mb, z=>sample_z(mb_size, z_dim)))
        # if k==5
        #     write(writer, dsum_curr, it)
        # end
    end
    _, Gl = run(sess,[G_solver, G_loss], feed_dict=
            Dict(z=>sample_z(mb_size, z_dim)))
    @show it, Dl, Gl

    push!(_gloss, Gl)
    push!(_dloss, Dl)
end
