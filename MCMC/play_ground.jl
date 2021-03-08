using PyPlot 
using AdFem


mmesh = Mesh(10,10,0.1)
A = compute_fem_laplace_matrix1(mmesh)
F = eval_f_on_gauss_pts((x, y)->2π^2 * sin(π*x) * sin(π*y), mmesh)
rhs = compute_fem_source_term1(F, mmesh)
bdnode = bcnode(mmesh)
A, rhs = impose_Dirichlet_boundary_conditions(A, rhs, bdnode, zeros(length(bdnode)))
sol = A\rhs 

# sess = Session(); init(sess)
# S = run(sess, sol)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(sol, mmesh)
title("Computed")
subplot(122)
visualize_scalar_on_fem_points(eval_f_on_fem_pts((x,y)->sin(π*x)*sin(π*y), mmesh), mmesh)
title("Reference")
savefig("possion.png")