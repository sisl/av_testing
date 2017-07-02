using Distributions
using Parameters
using PGFPlots

@with_kw type ProblemDefinition
    H::Int = 2 # horizon, the number of steps
    β_transitions::Vector{Int} = [-1,0,1,2]
    β_distribution::Vector{Float64} = [0.1,0.3,0.5,0.1]

    # Problem specs
    λ_ref::Float64 = 1.0
    C_ref::Float64 = 0.95
    rho::Float64 = 0.9
    gamma::Float64 = 1.0

    # Parameters for relevant experience and innovation
    var::Float64 = 1.0
    μ::Float64 = 1.0
    prob_k = Dict{Int, Vector{Float64}}()
    conditional_probs = Dict{Vector{Any}, Float64}()
end

immutable State
    α::Float64
    β::Float64
    innov::Int
end
Base.show(io::IO, s::State) = print(io, "State(", s.α, ",", s.β, ",", s.innov, ")" )

typealias Action Int
Base.next(s::State, a::Action, k::Int, Δβ::Int) = State(s.α + k, s.β + a, s.innov + Δβ)

typealias PolicySlice Dict{State, Action}
typealias Policy Vector{Dict{State, Action}}
typealias UtilitySlice Dict{State, Float64}
typealias Utility Vector{Dict{State, Float64}}

# Find probability of being in terminal state
function P_terminal(s::State, λ::Float64)
    d = Gamma(s.α, 1/s.β)
    return cdf(d, λ)
end
P_terminal(s::State, def::ProblemDefinition) = P_terminal(s, def.λ_ref)
function is_terminal(s::State, λ_ref::Float64, C_ref::Float64)
    conf = P_terminal(s, λ_ref)
    return conf > C_ref
end
is_terminal(s::State, def::ProblemDefinition) = is_terminal(s, def.λ_ref, def.C_ref)

# Find probability of reaching terminal state by releasing n vehicles
function P_n(s::State, n::Int, def::ProblemDefinition)
    prob = 0.0
    k = 0
    P = NegativeBinomial(n*s.α, (s.β + s.innov)/(1+s.β + s.innov))
    while is_terminal(next(s, n, k, 0), def)
        prob += pdf(P, k)
        k += 1
    end
    return prob
end

# Solve for expected utility for future states from a given state
function future_utility(s::State, n::Action, U::Utility, π::Policy, h::Int, def::ProblemDefinition, Δβ::Int=0)
    utility = 0.0

    # No utility for terminal state
    if is_terminal(s, def)
        return utility
    end

    # Construct distribution over adverse events
    P = NegativeBinomial(n*s.α, (s.β + s.innov)/(1+s.β + s.innov))
    k = 0

    # Find utility of next state if not already known
    next_state = next(s, n, k, Δβ)
    if !(next_state in keys(U[h]))
        U[h][next_state], _ = find_u_n(next_state, U, π, h, def)
    end

    # Find utility of future states
    while ((U[h][next_state] > 0.0  || is_terminal(next(s, n, k, Δβ), def)))
        utility += U[h][next_state]*pdf(P, k)

        # Update next state
        k += 1
        next_state = next(s, n, k, Δβ)
        if !(next_state in keys(U[h]))
            U[h][next_state], _ = find_u_n(next_state, U, π, h, def)
        end
    end
    utility
end

# Find utility and optimal policy for belief state
function find_u_n(s::State, U::Utility, π::Policy, h::Int, def::ProblemDefinition)
    if is_terminal(s, def)
        u = 0.0
        n = -1
    else
        # Calculate utility for each value of n
        n_max = min(10, Int(floor(def.rho/(1 - def.rho)*(s.β + s.innov)/s.α)))
        utilities = zeros(n_max+1)
        for n = 0:n_max
            # Initialize with penalty for collisions
            utilities[n+1] += -(1 - def.rho)*n*s.α/(s.β + s.innov)

            # Reward for reaching terminal state
            if is_terminal(next(s, n, 0, 0), def)
                utilities[n+1] += def.rho*P_n(s, n, def)
            end
            
            for i = 1:length(def.β_transitions)
                # Don't allow transition to infeasible belief state, have bonus scale with num of vehicles
                if (n + s.β + n*def.β_transitions[i]) <= 0
                    Δβ = 1-n-s.β
                else
                    Δβ = n*def.β_transitions[i]
                end
                
                # Utility for future states
                if h < def.H
                    if n == 0
                        if !(s in keys(U[h+1]))
                            U[h+1][s], _ = find_u_n(s, U, π, h+1, def)
                        end
                        utilities[n+1] += def.β_distribution[i]*def.gamma*U[h+1][s]
                    else 
                        utilities[n+1] += def.β_distribution[i]*def.gamma*future_utility(s, n, U, π, h+1, def, Δβ)
                    end
                end
            end
        end
        u = maximum(utilities)
        n = indmax(utilities) - 1
    end
    return u, n
end

# Solve for utility and policy at given timestep
function find_policy!(states::Set{State}, U::Utility, π::Policy, h::Int, def::ProblemDefinition)
    for s in states
        u, n = find_u_n(s, U, π, h, def)
        π[h][s] = n
        U[h][s] = u
    end
end

# Convert policy dict to array
function array_policy(π::PolicySlice, α_prior::Float64, β_prior::Float64, α_lo::Int, α_hi::Int, β_lo::Int, β_hi::Int)
    policy = zeros(α_hi - α_lo + 1, β_hi - β_lo + 1)
    for i = α_lo : α_hi
        for j = β_lo : β_hi
            s = State(i + α_prior, j + β_prior, 0)
            policy[i, j] = π[s]
        end
    end
    policy
end

# Solve finite-horzon MDP given problem definition
function solve(α_lo::Int, α_hi::Int, β_lo::Int, β_hi::Int, def::ProblemDefinition)
    # Find prior belief based on problem definition
    β_prior = def.μ/def.var
    α_prior = def.μ*β_prior

    # Construct state space to compute utility
    states = Set{State}()
    for α in α_lo : α_hi
        for β in β_lo : β_hi
            push!(states, State(α + α_prior, β + β_prior, 0))
        end
    end

    # Initialize policy and set of utilities
    π = Array(PolicySlice, def.H)
    U = Array(UtilitySlice, def.H)

    # Loop back through time and compute values
    for h in def.H : -1 : 1
        println("horizon: $h")
        π[h] = PolicySlice()
        U[h] = UtilitySlice()
        find_policy!(states, U, π, h, def)
    end

    # Create and return policy array
    policies = zeros(def.H, α_hi, β_hi)
    for h in 1:def.H
        policies[h, :, :] = array_policy(π[h], α_prior, β_prior, α_lo, α_hi, β_lo, β_hi)
    end
    policies
end
