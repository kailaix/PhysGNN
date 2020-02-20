for hidden_size in 2 10 20
do 
for n_layer in 3 8 32
do 
for model_id in 1 2 3 4
do 
julia Gauss.jl $hidden_size $n_layer $model_id & 
done
done
done
wait