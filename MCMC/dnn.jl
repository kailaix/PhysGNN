include("common.jl")
reset_default_graph()
sess = Session(); init(sess)

env = Environment(sess)
sim = Simulator(sess)

loss = empirical_sinkhorn(env.obs, sim.obs, method = "lp")

opt = AdamOptimizer(0.001).minimize(loss)
init(sess)
@info run(sess, loss, env.μ=>sample_exact(env),
    sim.μ=>sample_latent(sim))

db = ResultSet(overwrite = true)
for i = 1:1000
    _, l = run(sess, [opt, loss], env.μ=>sample_exact(env),
                     sim.μ=>sample_latent(sim))

    if mod(i, 50)==0
        for k = 1:100
            d = run(sess, sim.dnn, sim.μ=>sample_latent(sim))
            save_to_db(db, i, l, d)
        end

        @info i, l  
    end
end
commit(db)
close(db)
