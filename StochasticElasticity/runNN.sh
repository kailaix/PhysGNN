for tid in 0 1 2 3 4 5 6 7 8 9
do 
for hmat_idx in 1 2 3
do 
srun -n 1 -N 1 --partition=CPU julia nn.jl $tid $hmat_idx
done 
done 