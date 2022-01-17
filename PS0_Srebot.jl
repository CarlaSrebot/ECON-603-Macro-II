#=======================
    ECON 603 - PS0 

      Carla Srebot
========================#

# Question 1 - Tauchen's method 
using Distributions

## Defining Tauchen's function
function tauchen(N::Integer, ρ::T1, σ::T2, m::Integer=3) where {T1 <: Real, T2 <: Real}
    # Get discretized space
    a_bar = m * sqrt(σ^2 / (1 - ρ^2))
    y_bar = range(-a_bar, stop=a_bar, length=N)
    w = y_bar[2] - y_bar[1]

    # Get transition probabilities
    Π = zeros(promote_type(T1, T2), N, N)
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

    # random initial value from y_var
    y = zeros(N)
    y_init = rand(y_bar)
    idx = findall(x->x == y_init,y_bar)
    
    prob_j = Π[idx, :]

    return prob_j
end

a = tauchen(4,0.2,0.4)



# Question 2