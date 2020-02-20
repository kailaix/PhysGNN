using Pkg
Pkg.add(["ADCME","Conda","MAT","JLD2","Revise","Distributions",
        "DelimitedFiles","LinearAlgebra","Statistics","PyCall", "RollingFunctions"])
using Conda,MAT,ADCME,JLD2,Revise,Distributions,DelimitedFiles,LinearAlgebra,Statistics,PyCall, RollingFunctions

Conda.add("scikit-learn")
function compile(DIR)
    PWD = pwd()
    cd(DIR)
    rm("build", force=true, recursive=true)
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd(PWD)
end

compile("$(@__DIR__)/StochasticElasticity/DirichletBD")
compile("$(@__DIR__)/StochasticElasticity/PlaneStressHmat")
compile("$(@__DIR__)/StochasticElasticity/SpatialFemStiffness")
compile("$(@__DIR__)/Poisson/ThomasOp")
compile("$(@__DIR__)/JumpDiffusion/jump")