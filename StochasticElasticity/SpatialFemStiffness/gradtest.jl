using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
# Random.seed!(233)

function spatial_fem_stiffness(hmat,m,n,h)
    spatial_fem_stiffness_ = load_op_and_grad("./build/libSpatialFemStiffness","spatial_fem_stiffness", multiple=true)
    hmat,m,n,h = convert_to_tensor([hmat,m,n,h], [Float64,Int32,Int32,Float64])
    spatial_fem_stiffness_(hmat,m,n,h)
end

# TODO: specify your input parameters
m = n = 20
h = 0.1
D = diagm(0=>[1,1,0.5])
Ds = zeros(m*n, 3, 3)
for i = 1:m 
    for j = 1:n 
        Ds[(j-1)*m+i,:,:] = D
    end
end
u = spatial_fem_stiffness(Ds,m,n,h)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m_)
    return sum(spatial_fem_stiffness(m_,m,n,h)[3]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(Ds)
v_ = rand(m*n,3,3)
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
