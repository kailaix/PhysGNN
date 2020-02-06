for tid in 1
do 
for hmat_idx in 1 2 3
do 
julia nn.jl $hmat_idx $tid &
# julia gs.jl $tid $hmat_idx &
done 
# wait %1 %2 %3 %4 %5 %6
done 