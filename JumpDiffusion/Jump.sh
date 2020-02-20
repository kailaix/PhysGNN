for i in 1 2 3
do 
julia jump2d.jl $i &
done 
wait

for i in 1 2 3
do 
julia learn2d.jl $i &
done 
wait