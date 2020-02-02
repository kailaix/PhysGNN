using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function plane_stress_hmat(e,nu)
    plane_stress_hmat_ = load_op_and_grad("./build/libPlaneStressHmat","plane_stress_hmat")
e,nu = convert_to_tensor([e,nu], [Float64,Float64])
    plane_stress_hmat_(e,nu)
end

# TODO: specify your input parameters
e = ones(10)
nu = 0.35*ones(10)
E = 1.0
ν = 0.35
HH = [
    E/(1-ν^2) ν*E/(1-ν^2) 0.0
    ν*E/(1-ν^2) E/(1-ν^2) 0.0
    0.0 0.0 E/(2(1+ν))
]
u = plane_stress_hmat(e,nu)
sess = Session(); init(sess)
@show run(sess, u)[end,:,:] - HH

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m_)
    return sum(plane_stress_hmat(e,m_)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10))*0.5
v_ = rand(10)*0.5
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

sess = Session(); init(sess)
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
