using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
matplotlib.use("agg")
Random.seed!(233)

jump_diffusion = load_op_and_grad("./build/libJumpDiffusion","jump_diffusion")

# TODO: specify your input parameters
mm = 10
n = 100
y = randn(mm)
tau = rand(mm)
t = 1.0
a = rand(mm+n+1)
b = rand(mm+n+1)
bp = rand(mm+n+1)
c = rand(mm+n+1)
dw = randn(mm+n+1)

y = constant(y)
tau = constant(tau)
t = constant(t)
n = constant(n, dtype = Int32)
mm = constant(mm, dtype = Int32)
a = constant(a)
b = constant(b)
bp = constant(bp)
c = constant(c)
dw = constant(dw)

u = jump_diffusion(y,tau,t,n,mm,a,b,bp,c,dw)
sess = tf.Session()
init(sess)
@show U = run(sess, u)

# error()
# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(jump_diffusion(y,tau,t,n,mm,a,b,m,c,dw))^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(10))
# v_ = rand(10)

m_ = constant(rand(111))
v_ = rand(111)


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
sval_ = abs.(run(sess, s_))
wval_ = abs.(run(sess, w_))
close("all")
loglog(gs_, sval_, "*-", label="finite difference")
loglog(gs_, wval_, "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
