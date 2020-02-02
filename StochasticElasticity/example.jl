include("CommonFuncs.jl")

# # example 1:
# m = 4
# n = 2
# h = 0.1
# Hs = zeros(32,m*n,3,3)
# for i = 1:32
#     Hs[i,:,:,:] = get_random_mat2()
# end
# # u = get_disp(H, m, n, h; tscale = 0.3)
# u = map(H->get_disp(H, m, n, h; tscale = 0.3), constant(Hs))
# sess = Session(); init(sess)
# u_ = run(sess, u)
# scatter(u_[3,:], m, n, h)


# # example 2:
# m = 10
# n = 5
# h = 0.1
# e = rand(m*n) .+ 0.5
# nu = rand(m*n)*0.5
# H = plane_stress_hmat(e, nu)
# u = get_disp(H, m, n, h; tscale = 0.3)
# sess = Session(); init(sess)
# u_ = run(sess, u)
# scatter(u_, m, n, h)

# # example 3
# m = 4
# n = 2
# h = 0.1
# z = placeholder(Float64, shape=[16*m*n,10])
# _, H = ae_Hmat(z, [20,20,20,2])
# H = tf.reshape(H, (16, m*n, 3, 3))
# u = get_disps(H, m, n, h)
# sess = Session(); init(sess)
# u_ = run(sess, u, z=>randn(16*m*n,10))
# scatter(u_[4,:], m, n, h)

