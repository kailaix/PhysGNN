include("CommonFuncs.jl")

hmat_idx = 1

tloss = []
loss = []
for i = 0:9
    @info i 
    @load "nn$hmat_idx$i.jld2" res1 
    push!(tloss, res1.tloss)
    push!(loss, res1.loss)
    global vs = res1.ii
end

tloss = hcat(tloss...)
loss = hcat(loss...)

nn_tloss_ = []
nn_loss_ = []
nn_tloss_v = []
nn_loss_v = []
for i = 1:length(vs)
    push!(nn_tloss_, mean(tloss[i,:]))
    push!(nn_tloss_v, std(tloss[i,:]))
    push!(nn_loss_, mean(loss[i,:]))
    push!(nn_loss_v, std(loss[i,:]))
end



tloss = []
loss = []
for i = 0:9
    @info i 
    @load "gs$hmat_idx$i.jld2" res1 
    push!(tloss, res1.tloss)
    push!(loss, res1.loss)
    global vs = res1.ii
end

tloss = hcat(tloss...)
loss = hcat(loss...)

gs_tloss_ = []
gs_loss_ = []
gs_tloss_v = []
gs_loss_v = []
for i = 1:length(vs)
    push!(gs_tloss_, mean(tloss[i,:]))
    push!(gs_tloss_v, std(tloss[i,:]))
    push!(gs_loss_, mean(loss[i,:]))
    push!(gs_loss_v, std(loss[i,:]))
end


close("all")
semilogy(vs, nn_tloss_, "r")
fill_between(vs, nn_tloss_ - nn_tloss_v, nn_tloss_+nn_tloss_v, alpha=0.5, color="red")

semilogy(vs, gs_tloss_, "g")
fill_between(vs, gs_tloss_ - gs_tloss_v, gs_tloss_+gs_tloss_v, alpha=0.5, color="green")

# semilogy(vs, loss_, "--", color="orange")
# fill_between(vs, loss_ - loss_v, loss_+loss_v, alpha=0.5, color="orange")
savefig("test.png")