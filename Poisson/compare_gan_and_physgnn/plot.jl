using PyPlot
using DelimitedFiles
using Distributions
matplotlib.use("agg")
using PyCall
mpl = pyimport("tikzplotlib")


dist0 = Dirichlet(3, 1.0)
dist1 = Dirichlet([1.0;2.0;3.0])

function plot_dist()
    V = rand(dist0, 10000)
    x, y = V[1,:], V[2,:]
    close("all")
    hist2D( x, y, normed=true, 
                bins=50, range=((0.0,1.0),(0.0,1.0)), vmin=0.0, vmax=5.0)
    colorbar()
    savefig("hist1.jpg")

    V = rand(dist1, 10000)
    x, y = V[1,:], V[2,:]
    close("all")
    hist2D( x, y, normed=true, 
                bins=50, range=((0.0,1.0),(0.0,1.0)), vmin=0.0, vmax=12.0)
    colorbar()
    savefig("hist2.jpg")
end


function plot_loss_ana(dir)
    close("all")
    loss = readdlm(joinpath(dir, "loss.txt"))[1:50000,:]
    plot(1:500:50000, loss[1:500:50000,1], label="Generator")
    plot(1:500:50000, loss[1:500:50000,2], label="Discriminator")
    legend()
    xlabel("Iteration")
    ylabel("Loss")
    grid("on", which = "both")
    savefig("loss$dir.pdf", bbox_inches = "tight")
end

function plot_loss_nnmc(dir)
    close("all")
    loss = readdlm(joinpath(dir, "loss.txt"))[1:3000,:]
    tloss = readdlm(joinpath(dir, "tloss.txt"))[1:16,:]
    plot(1:50:3000, loss[1:50:3000], label="Train")
    plot(1:200:3200, tloss, label="Validation")
    yscale("log")
    legend()
    xlabel("Iteration")
    ylabel("Loss")
    grid("on", which = "both")
    savefig("loss$dir.pdf", bbox_inches = "tight")
end


plot_loss_ana("analogs1")
plot_loss_ana("analogs2")
plot_loss_nnmc("nnmclogs1")
plot_loss_nnmc("nnmclogs2")

plot_dist()