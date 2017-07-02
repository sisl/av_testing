using JLD
include("finite_horizon.jl")

# Define vectors of parameter values
rhos = collect(0.75:0.1:0.95)
gammas = collect(0.5:0.25:1.0)
vars = [0.1, 1.0]
mus = [0.0, 0.5, 1.0]
probs = Set{Vector{Float64}}()
probs = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.25, 0.25, 0.5], [0.0, 0.0, 1.0, 0.0], [0.25, 0.5, 0.25, 0.0], 
            [0.0, 0.75, 0.0, 0.25], [0.25, 0.25, 0.25, 0.25], [0.0, 0.5, 0.25, 0.25]]

# Initialize dict mapping from params to policy
params_policy_dict = Dict{Array{Any, 1}, Array{Float64, 3}}()
for r in rhos
    for g in gammas
        for p in probs
            for v in vars
                for m in mus
                    println("r: $r, γ: $g, probs: $p, var: $v, mu: $m")
                    def = ProblemDefinition(H = 5, rho=r, C_ref=0.95, gamma = g, β_distribution = p, μ = m, var = v)
                    π = solve(1, 50, 1, 50, def)
                    params_policy_dict[[r, g, p, v, m]] = π
                end
            end
        end
    end
end
jldopen("param_sweep_full.jld", "w") do file
    write(file, "policies", params_policy_dict)
end
