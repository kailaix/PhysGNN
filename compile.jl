using ADCME
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