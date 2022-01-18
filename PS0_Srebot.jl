#=======================
    ECON 603 - PS0 

      Carla Srebot
========================#
using Distributions, Random, StatsBase
Random.seed!(101)


# Question 1 - Tauchen's method 

## Defining Tauchen's function
function tauchen(N::Integer, ρ::Real, σ::Real, m::Integer=3) 
    # Get discretized space
    a_bar = m * sqrt(σ^2 / (1 - ρ^2))
    y_bar = collect(range(-a_bar, stop=a_bar, length=N))
    w = y_bar[2] - y_bar[1]

    # Get transition probabilities
    Π = zeros(N, N)
    for row = 1:N
        # Do end points first
        Π[row, 1] = cdf.(Normal(), (y_bar[1] - ρ*y_bar[row] + w/2) / σ)
        Π[row, N] = 1 - cdf.(Normal(), (y_bar[N] - ρ*y_bar[row] - w/2) / σ)

        # fill in the middle columns
        for col = 2:N-1
            Π[row, col] = (cdf.(Normal(),(y_bar[col] - ρ*y_bar[row] + w/2) / σ) - cdf.(Normal(),(y_bar[col] - ρ*y_bar[row] - w/2) / σ))
        end
    end

    # renormalize the transition matrix: 
    Π = Π./sum(Π, dims = 2)

    return (Π, y_bar)
end


# Question 2 - Rouwenhorst's method 
function rouwenhorst(p::Real, q::Real, ψ::Real, n::Integer)
    if n == 2
        return [-ψ, ψ],  [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst(p, q, ψ, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return (θN, collect(range(-ψ, stop=ψ, length=n)))
    end
end


# Question 3
per = 1000
N = 3

ρ = 0.2
σ = 0.4


## Tauchen's method 
    t1 = tauchen(N,ρ,σ)

    # random initial value from y_var
    y_tauchen = zeros(per)
    y_tauchen_init = rand(t1[2])
    idx_tauchen_init = findall(x->x == y_tauchen_init,t1[2])

    # defining Markov process
    y_tauchen[1] = sample(t1[2], Weights(vec(t1[1][idx_tauchen_init, :])))

    for col = 2:per
        idx = findall(x->x == y_tauchen[col-1],t1[2])
        y_tauchen[col] = sample(t1[2], Weights(vec(t1[1][idx, :])))
    end 

    print(y_tauchen)


## Rouwenhorst's method 
    p = (1+ρ)/2
    q = (1+ρ)/2
    ψ = sqrt(((N-1)*σ^2)/(1-ρ^2))
    
    r1 = rouwenhorst(p, q, ψ, N)
    
    # random initial value from y_var
    y_rouwenhorst = zeros(per)
    y_rouwenhorst_init = rand(r1[2])
    idx_rouwenhorst_init = findall(x->x == y_rouwenhorst_init, r1[2])

    # defining Markov process
    y_rouwenhorst[1] = sample(r1[2], Weights(vec(r1[1][idx_rouwenhorst_init, :])))

    for col = 2:per
        idx = findall(x->x == y_rouwenhorst[col-1],r1[2])
        y_rouwenhorst[col] = sample(r1[2], Weights(vec(r1[1][idx, :])))
    end 

    print(y_rouwenhorst)
    