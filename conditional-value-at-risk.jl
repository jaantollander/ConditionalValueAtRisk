"""Value-at-risk."""
function value_at_risk(x::Vector{Float64}, f::Vector{Float64}, α::Float64)
    i = findfirst(p -> p≥α, cumsum(f))
    if i === nothing
        return x[end]
    else
        return x[i]
    end
end

"""Conditional value-at-risk."""
function conditional_value_at_risk(x::Vector{Float64}, f::Vector{Float64}, α::Float64)
    x_α = value_at_risk(x, f, α)
    if iszero(α)
        return x_α
    else
        tail = x .≤ x_α
        return (sum(x[tail] .* f[tail]) - (sum(f[tail]) - α) * x_α) / α
    end
end
