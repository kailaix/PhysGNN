for tid in 0 1 2 3 4 5 6 7 8 9
do 
for hmat_idx in 1 2 3
do 
julia nn.jl $tid $hmat_idx &
julia gs.jl $tid $hmat_idx &
done 
wait %1 %2 %3 %4 %5 %6
done 