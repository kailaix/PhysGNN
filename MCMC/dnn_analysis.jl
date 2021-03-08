include("common.jl")



db = Database("result.db")
conn = execute(db, """
SELECT * FROM exact
""")
result = collect(conn)
mu = [x[1] for x in result]
obs = [x[2] for x in result]

conn = execute(db, """
SELECT * FROM mcmc
""")
result = collect(conn)
mu_mcmc = [x[1] for x in result]
obs_mcmc = [x[2] for x in result]


x0 = reshape(collect(LinRange(0, 2, 100)), :, 1)
kde = fit_kde(mu, 0.1)
exc = exp.(kde(x0))

kde = fit_kde(mu_mcmc, 0.1)
exc_mcmc = exp.(kde(x0))
make_directory("dnn_figures")

for i = 50:50:1000
    conn = execute(db, """
SELECT loss, dnn from result WHERE iter = $i 
""")
    res = collect(conn)
    result = [x[2] for x in res]
    @info res[1][1]
    close("all")

    kde = fit_kde(result, 0.1)
    out = exp.(kde(x0))
    plot(x0, exc, label = "Exact")
    plot(x0, exc_mcmc, label = "MCMC")
    plot(x0, out, label = "PhysGNN")
    xlabel("\$\\mu\$", fontsize = 15)
    ylabel("Density", fontsize = 15)
    legend(fontsize = 15)
    tight_layout()
    
    savefig("dnn_figures/result$i.png")
    savefig("dnn_figures/result$i.pdf")
end
