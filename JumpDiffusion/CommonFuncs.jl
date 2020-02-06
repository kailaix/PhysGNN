
using ADCME
using Distributions

function make_data(λ, T, N, Ns=1000; max_size=nothing, a=missing, b=missing, bp=missing, c=missing)
    d = Exponential(λ)
    a = coalesce(a, (s, t)->0.0)
    b = coalesce(b, (s, t)->0.0)
    bp = coalesce(bp, (s, t)->0.0)
    c = coalesce(c, (s,t)->1.0)
    get_t = ()->begin
        τ = Float64[]
        t = 0.0
        while true
            Δt = rand(d)
            t += Δt
            if t<T 
                push!(τ, t)
            else
                break
            end
        end
        τ
    end
    ts = []
    while length(ts)!=Ns
        tt = get_t()
        if !isnothing(max_size) && length(tt)>max_size
            @warn "check max_size"
            continue
        end
        push!(ts, tt)
    end

    maxval = maximum([length(x) for x in ts])
    if !isnothing(max_size)
        maxval = max_size
    end
    tau = zeros( Ns, maxval)
    ms = Int32[]
    for i = 1:Ns
        tau[i, 1:length(ts[i])] = ts[i]
        push!(ms, length(ts[i]))
    end
    t = ones(Ns)
    n = ones(Int32, Ns) * N

    t0 = [ones(Ns, 1)*(LinRange(0, T, N+1)|>collect)[:]' tau]
    a = a.(nothing, t0)
    b = b.(nothing, t0)
    bp = bp.(nothing, t0)
    c = c.(nothing, t0)

    dw = randn(Ns,maxval+N+1)

    tau, ms, t, n, a, b, bp, c, dw
end

function jdsim2(ys,tau,t,n,ms,a,b,bp,c,dw)
    n_ = n
    ys = convert_to_tensor(ys, dtype=Float64)
    tau = convert_to_tensor(tau, dtype=Float64)
    t = convert_to_tensor(t, dtype=Float64)
    n = convert_to_tensor(n, dtype=Int32)
    ms = convert_to_tensor(ms, dtype=Int32)
    a = convert_to_tensor(a, dtype=Float64)
    b = convert_to_tensor(b, dtype=Float64)
    bp = convert_to_tensor(bp, dtype=Float64)
    c = convert_to_tensor(c, dtype=Float64)
    dw = convert_to_tensor(dw, dtype=Float64)
    jump_diffusion = load_op_and_grad("./Jump/build/libJumpDiffusion","jump_diffusion")
    jump_diff(o) = jump_diffusion(o...)
    S = map(jump_diff, [ys,tau,t,n,ms,a,b,bp,c,dw], dtype=Float64)
    S.set_shape((size(dw,1), n_[1]+1))
    S
end


function make_data2(;λ=0.1, T=1.0, N=100, Ns=1000, max_size=nothing, cfun=nothing)
    if isnothing(cfun)
        cfun = t->1.0
    end
    d = Exponential(λ)
    get_t = ()->begin
        τ = Float64[]
        t = 0.0
        while true
            Δt = rand(d)
            t += Δt
            if t<T 
                push!(τ, t)
            else
                break
            end
        end
        τ
    end
    ts = []
    while length(ts)!=Ns
        tt = get_t()
        if !isnothing(max_size) && length(tt)>max_size
            @warn "check max_size"
            continue
        end
        push!(ts, tt)
    end

    maxval = maximum([length(x) for x in ts])
    if !isnothing(max_size)
        maxval = max_size
    end
    tau = zeros( Ns, maxval)
    ms = Int32[]
    for i = 1:Ns
        tau[i, 1:length(ts[i])] = ts[i]
        push!(ms, length(ts[i]))
    end
    t = ones(Ns)*T
    n = ones(Int32, Ns) * N
    c = ones(Ns, maxval+N+1)
    for i = 1:N+1
        c[:,i] *= cfun((i-1)*T/N)
    end
    for k = 1:Ns
        for i = 1:maxval
            c[k,i+N+1] *= cfun(tau[k, i])
        end
    end
    tau, ms, t, n, c
end


function jdsim3(ys,tau,t,n,ms,c)
    n_ = n
    ys = convert_to_tensor(ys, dtype=Float64)
    tau = convert_to_tensor(tau, dtype=Float64)
    t = convert_to_tensor(t, dtype=Float64)
    n = convert_to_tensor(n, dtype=Int32)
    ms = convert_to_tensor(ms, dtype=Int32)
    c = convert_to_tensor(c, dtype=Float64)
    multivariate_jump_diffusion = load_op_and_grad("./Jump/build/libJumpDiffusion","multivariate_jump_diffusion")
    multivariate_jump_diffusion_(o) = multivariate_jump_diffusion(o...)
    S = map(multivariate_jump_diffusion_, [ys, tau, t, n, ms, c], dtype=Float64)
    S.set_shape((length(ms), 2, n_[1]+1))
    S
end


function get_weights()
    vs = get_collection()
    weights = []
    for v in vs
        if occursin("weights", v.name)
            push!(weights,v)
        end
    end
    weights
end