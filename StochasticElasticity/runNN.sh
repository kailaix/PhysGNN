for tid in 0 1 2 3 4
do 
for hmat_idx in 1 2 3
do 
for latent_dim in 10 20 40 80 160
do 
julia nn.jl $hmat_idx $tid $latent_dim &
# julia gs.jl $tid $hmat_idx &
done 
wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15
done 
done 