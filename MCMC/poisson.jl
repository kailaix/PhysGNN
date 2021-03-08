using AdFem
using PyPlot 

mmesh = Mesh(10,10,0.1)


z = randn(1,10)
dnn = fc(z, [20,20,20,1])|>squeeze 
κ = (1+dnn) * ones(get_ngauss(mmesh))
A = compute_fem_laplace_matrix1(κ, mmesh)
F = eval_f_on_gauss_pts((x, y)->2π^2 * sin(π*x) * sin(π*y), mmesh)
rhs = compute_fem_source_term1(F, mmesh)
bdnode = bcnode(mmesh)
A, rhs = impose_Dirichlet_boundary_conditions(A, rhs, bdnode, zeros(length(bdnode)))
sol = A\rhs 

sess = Session(); init(sess)

SOL = run(sess, sol)
close("all")
visualize_scalar_on_fem_points(SOL, mmesh)
savefig("poisson-dnn.png")