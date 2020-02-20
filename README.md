# PhysGNN: learning generative neural network with physics knowledge

## Instruction 


Run the following command in **current directory** 
```julia
include("compile.jl")
```
⚠️ If it is the first time you run the command above, it might take you some time for downloading and installing dependencies. 

Now you can run any script in this repository. 


## Detailed Instruction

:warning: Please disable GPU for the following examples: 

This section provides detailed intructions to reproduce the examples in the paper

### MNIST Example

```julia
cd("MNIST")
include("DCGAN.jl") # DCGAN
include("PhysGNN.jl") # PhysGNN
include("SHGAN1.jl") # Sinkhorn GAN with the penalty parameter = 1.0
include("SHGAN10.jl") # Sinkhorn GAN with the penalty parameter = 10.0
include("SHGAN100.jl") # Sinkhorn GAN with the penalty parameter = 100.0
include("SHGAN10000.jl") # Sinkhorn GAN with the penalty parameter = 10000.0

```



