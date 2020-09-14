using Random, Plots
include("plotting.jl")

rng = MersenneTwister(1)
x, f = random_probability_distribution(rng, 3, 5, -2.0, 3.0)
α = 0.25

@assert issorted(x)
@assert all(f .≥ 0)
@assert sum(f) ≈ 1

dir = "images"

@info "Distributions"
plt1 = plot_distributions(x, f, α)
savefig(plt1, joinpath(dir, "distributions.svg"))

@info "VaR"
plt2 = plot_VaR(x, f, α)

@info "CVaR"
plt3 = plot_CVaR(x, f)

plt4 = plot(plt2, plt3, layout=(1, 2), legend=false, size=(720, 400))
savefig(plt4, joinpath(dir, "conditional-value-at-risk.svg"))
