for hmat_idx in 1 2 3
do
julia gs.jl $hmat_idx &
# julia gs.jl $tid $hmat_idx &
done 