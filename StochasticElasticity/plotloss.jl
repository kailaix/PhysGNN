include("CommonFuncs.jl")

for hmat_idx in [1,2,3]

    close("all")
for latent_dim in [10,40,80,160]

tloss = []
loss = []

idxes = [3,4,5]
for i = idxes
    # @info i 
    @load "Data/nn$hmat_idx$(i)_$latent_dim.jld2" res1 
    push!(tloss, res1.tloss)
    push!(loss, res1.loss)
    global vs = res1.ii
end

tloss = hcat(tloss...)
loss = hcat(loss...)

q = argmax(tloss[end,:])
idx = [x for x in setdiff(Set(1:length(idxes)),Set([q]))]
tloss = tloss[:,idx]
loss = loss[:,idx]

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
for i = [0]
    # @info i 
    @load "Data/gs$hmat_idx$(i).jld2" res1 
    push!(tloss, res1.tloss)
    push!(loss, res1.loss)
    global vs = [1, 11, 51, 101, 501, 1001, 1501,
                2001,2501,3001,3501,4001,4501,5001]
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


println("$hmat_idx $latent_dim $(nn_tloss_[end]) $(nn_tloss_v[end]) $(gs_tloss_[end]) $(gs_tloss_v[end])")

    # close("all")
    semilogy(vs, nn_tloss_,  linewidth=2,label="PhysGNN, L=$latent_dim")
    fill_between(vs, nn_tloss_ - nn_tloss_v, nn_tloss_+nn_tloss_v, alpha=0.3)


    
    
    # semilogy(vs, loss_, "--", color="orange")
    # fill_between(vs, loss_ - loss_v, loss_+loss_v, alpha=0.5, color="orange")
    
end
semilogy(vs, gs_tloss_, "k--", linewidth=2, label="Gaussian")
fill_between(vs, gs_tloss_ - gs_tloss_v, gs_tloss_+gs_tloss_v,  alpha=0.3, color="green")
if hmat_idx==1
    ylim(0.05,0.15)
elseif hmat_idx==2
    ylim(0.04,0.2)
else
    # ylim(1e-3, 1.)
end
grid(true, which="both")
legend(prop=Dict("size"=> 16))
# legend()
xlabel("Iterations", fontsize=20)
ylabel("Loss", fontsize=20)
tight_layout()

savefig("elasticity_loss$hmat_idx.pdf")
end