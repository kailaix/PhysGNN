# PhysGNN: learning generative neural network with physics knowledge

## Instruction 

Run the following command in **current directory** to activate the environment and install necessary packages. 
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

If it is the first time you use this code, compile binary dependencies using 
```julia
include("compile.jl")
```

To run a script in command line, make sure you include the project flag
```bash
julia --project="<path to PhysGNN>" <script file name>
```

