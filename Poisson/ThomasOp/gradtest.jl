using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libSolveTridiagonal = tf.load_op_library('build/libSolveTridiagonal.so')
@tf.custom_gradient
def solve_tridiagonal(a,b,c,f):
    d = libSolveTridiagonal.solve_tridiagonal(a,b,c,f)
    def grad(dy):
        return libSolveTridiagonal.solve_tridiagonal_grad(dy, d, a,b,c,f)
    return d, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSolveTridiagonal = tf.load_op_library('build/libSolveTridiagonal.dylib')
@tf.custom_gradient
def solve_tridiagonal(a,b,c,f):
    d = libSolveTridiagonal.solve_tridiagonal(a,b,c,f)
    def grad(dy):
        return libSolveTridiagonal.solve_tridiagonal_grad(dy, d, a,b,c,f)
    return d, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSolveTridiagonal = tf.load_op_library('build/libSolveTridiagonal.dll')
@tf.custom_gradient
def solve_tridiagonal(a,b,c,f):
    d = libSolveTridiagonal.solve_tridiagonal(a,b,c,f)
    def grad(dy):
        return libSolveTridiagonal.solve_tridiagonal_grad(dy, d, a,b,c,f)
    return d, grad
"""
end

solve_tridiagonal = py"solve_tridiagonal"

# TODO: 
n = 100
a = constant(3*ones(n))
f = constant(ones(n))
b = constant(-ones(n-1))
c = constant(-ones(n-1))
u = solve_tridiagonal(a,b,c,f)
u.set_shape((n,)) # a bug in the cluster?
sess = Session()
init(sess)
uval = run(sess, u)
p = diagm(0=>3*ones(n), -1=>-ones(n-1), 1=>-ones(n-1))
sol = p\ones(n)
@show norm(uval-sol)
# TODO: 

# error("stop")
# gradient check -- v
function scalar_function(m)
    return sum(tanh(solve_tridiagonal(a,b,m,f)))
end

m_ = constant(rand(n-1))
v_ = rand(n-1)
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

plt[:gca]()[:invert_xaxis]()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
