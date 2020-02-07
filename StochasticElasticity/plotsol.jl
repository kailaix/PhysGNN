include("CommonFuncs.jl")
using Random; Random.seed!(233)

m = 4
n = 2
h = 0.1
Hs = zeros(32,m*n,3,3)
for i = 1:32
    Hs[i,:,:,:] = get_random_mat2()
end
# u = get_disp(H, m, n, h; tscale = 0.3)
u = map(H->get_disp(H, m, n, h; tscale = 0.3), constant(Hs))
sess = Session(); init(sess)
u_ = run(sess, u)
mesh(u_[1:6,:], m, n, h, facecolors = split("rgmbyc",""), edgecolors=true)
xlim(-h*0.1, m*h+h)
ylim(-0.3*h, n*h+0.3*h)
axis("off")
axis("equal")

savefig("fem.png")

close("all")
m = 8
n = 4
h = 0.05
Hs = zeros(6,m*n,3,3)
for i = 1:6
    Hs[i,:,:,:] = get_random_mat2()
end
# u = get_disp(H, m, n, h; tscale = 0.3)
u = map(H->get_disp(H, m, n, h; tscale = 0.3), constant(Hs))
sess = Session(); init(sess)
u_ = run(sess, u)
mesh(u_, m, n, h, facecolors = split("rgmbyc",""), edgecolors=false)
xlim(-h*0.1, m*h+2*h)
ylim(-0.3*h, n*h+0.3*h)
axis("off")
axis("equal")

savefig("obs.png")


close("all")
m = 40
n = 20
h = 0.01
Hs = zeros(32,m*n,3,3)
for i = 1:32
    Hs[i,:,:,:] = get_random_mat2()
end

p = mesh(zeros(2*(m+1)*(n+1)), m, n, h, edgecolors=true)
p.set_array(100*rand(m*n))

plot([0.0
      0.0
      m*h
      m*h
      0.0], [
        0.0
        n*h
        n*h
        0.0
        0.0
      ],"k", linewidth=2.0)
for i = 1:10
    x = m*h
    y = n*h*(i-0.5)/10
    arrow(x, y, 0.02, 0.0 ,width=0.005, head_length=h,
        edgecolor=nothing, color="k")
end

axis("equal")
xlim(-h*0.1, m*h+0.03)
ylim(-0.3*h, n*h+0.3*h)
axis("off")
savefig("physics.png")



# https://github.com/UniversityofWarwick/Brownian.jl
close("all")

for i = 1:20
    p = randn(100)
    p = cumsum(p)
    plot(p)
end
axis("off")
savefig("brownian.png")


function genrand(idx)
    if idx==1
        if rand()>0.5
            u = 0.05*randn()+0.75
            v = 0.05*randn()+0.75
        else
            u = 0.05*randn()+0.25
            v = 0.05*randn()+0.25
        end
        u + 1.0, v*0.5
    elseif idx==2
        u = 0.2*randn()
        u+ 1.5, 0.4+randn()*0.02
    elseif idx==3
        u,v = randn(2)
        u*0.05 + 1.7, v*0.05 + 0.3
    elseif idx==4
        
    end
end

for idx in [1 2 3]
    close("all")
    U = [genrand(idx) for i = 1:10000]
    hist2D([x[1] for x in U], [x[2] for x in U], bins=50, density=true, range=((1.0,2.0), (0.0,0.5)))
    colorbar()
    xlabel("E")
    ylabel("\$\\nu\$")
    savefig("$idx.pdf")
end



close("all")
pts = @. ([-1/sqrt(3); 1/sqrt(3)] + 1)/2
m = 4
n = 2
h = 0.1
plot([0.0
      0.0
      m*h
      m*h
      0.0], [
        0.0
        n*h
        n*h
        0.0
        0.0
      ],"k", linewidth=2.0)

xs = []
ys = []
for i = 1:m 
    for j = 1:n 
        plot([(i-1)*h; i*h; i*h; (i-1)*h; (i-1)*h], [(j-1)*h; (j-1)*h; j*h; j*h; (j-1)*h],
                    "k", linewidth=2.0)

        for p = 1:2
            for q = 1:2
                push!(xs, (i-1)*h + pts[p]*h)
                push!(ys, (j-1)*h + pts[q]*h)
            end
        end
    end
end
for i = 1:10
    x = m*h
    y = n*h*(i-0.5)/10
    arrow(x, y, 0.02, 0.0 ,width=0.005, head_length=0.2*h,
        edgecolor=nothing, color="k")
end
scatter(zeros(2*(m+1)*(n+1)), m, n, h, color="k")
scatter(xs, ys, marker="o", color="red")
axis("equal")
axis("off")

xlim(-h*0.3, m*h+0.05)
ylim(-0.3*h, n*h+0.3*h)
savefig("plate.pdf")