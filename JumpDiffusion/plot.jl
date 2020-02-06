using PyPlot
using DelimitedFiles
using Distributions
matplotlib.use("agg")
using PyCall
mpl = pyimport("tikzplotlib")

function plot_loss(i)
    close("all")
    loss = readdlm("2D$i/loss.txt")
    tloss = readdlm("2D$i/tloss.txt")
    plot(1:10:1000, loss[1:10:1000], label="Train")
    plot(1:100:1001, tloss[:], label="Validation")
    yscale("log")
    legend()
    xlabel("Iteration")
    ylabel("Loss")
    grid("on", which = "both")
    savefig("jumploss$i.pdf", bbox_inches = "tight")
end

plot_loss(1)
plot_loss(2)
plot_loss(3)