using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
matplotlib.use("agg")

multivariate_jump_diffusion = load_op_and_grad("./build/libJumpDiffusion","multivariate_jump_diffusion")

# TODO: specify your input parameters
y = rand(2,10)|>constant
tau = rand(10)|>constant 
t = constant(1.0)
n = constant(100, dtype=Int32)
m = constant(7, dtype=Int32)
c = constant(ones(111))
u = multivariate_jump_diffusion(y,tau,t,n,m,c)
sess = tf.Session()
init(sess)
@show run(sess, u)

# error()
# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(x)
    return sum(multivariate_jump_diffusion(x,tau,t,n,m,c)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(2,10))
v_ = rand(2,10)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session()
init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
