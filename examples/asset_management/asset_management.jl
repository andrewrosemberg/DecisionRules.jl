using AbstractGPs, KernelFunctions #, TemporalGPs
using Random

function generate_asset_returns_process(;num_assets = 10, num_factors = 3, noise_var = 0.1, 
    number_scenarios = 100, num_t=1000,
    A = randn(num_assets, num_factors) * 0.01
)
    x = range(-5.0; step=0.1, length=num_t)
    factors_naive = [GP(compose(PeriodicKernel(;r=[rand(0.001:0.005:10)]) * ExponentialKernel(), ScaleTransform(0.2))) for _ in 1:num_factors]

    # Project onto finite-dimensional distribution as usual.
    fx = [factors_naive[i](x, noise_var) for i in 1:num_factors]

    # Sample from the prior as usual.
    y = [rand(fx[i], number_scenarios) for i in 1:num_factors] # Samples organized by [Factor [Scenario, Time]]

    # Organoze the samples by [Time, Scenario, Factor]
    y = reshape(hcat(y...), (length(x), number_scenarios, num_factors))

    asset_prices =  [Matrix((A * y[:,s,:]')') for s in 1:number_scenarios]

    return factors_naive, y, asset_prices, x, A
end

function calculate_posterior_returns(factors_naive, x_observed, y_observed, x_predict, A; 
    num_scenarios = 100, noise_var = 0.1,
)
    num_factors = length(factors_naive)
    fx = [factors_naive[i](x_observed, noise_var) for i in 1:num_factors]
    p_fx = [posterior(fx[i], y_observed[:, i]) for i in 1:num_factors]

    y = [rand(p_fx[i](x_predict, noise_var), num_scenarios) for i in 1:num_factors]
    y = reshape(hcat(y...), (length(x_predict), num_scenarios, num_factors))

    asset_prices = [Matrix((A * y[:,s,:]')') for s in 1:num_scenarios]

    return y, asset_prices
end
num_t = 200
num_factors = 3
num_assets = 10
number_scenarios = 100
noise_var = 0.1
factors_naive, factors, asset_prices, x, A = generate_asset_returns_process(;num_assets=num_assets, num_factors=num_factors, noise_var=noise_var, number_scenarios=number_scenarios, num_t=num_t)

using Plots
plt = plot(factors[:,1,1], label="Factor 1")
for i in 2:num_factors
    plot!(plt, factors[:, 1, i], label="Factor $i")
end
plt

plt = plot(asset_prices[1][:,1:3], label="")

# Posterior returns at time t=100 for t=101:110
scen = 1
idx_observed = 1:100
x_observed = x[idx_observed]
y_observed = factors[idx_observed, scen, :]
idx_predict = 101:110
x_predict = x[idx_predict]
y_predict, asset_prices_predict = calculate_posterior_returns(factors_naive, x_observed, y_observed, x_predict, A)

# plot observed and predicted asset prices
plt = plot(idx_observed, asset_prices[scen][idx_observed,1], label="Observed", color=:blue)
plot!(plt, idx_predict, asset_prices_predict[1][:,1], label="Predict", color=:blue, alpha=0.1)
for i in 2:100
    plot!(plt, idx_predict, asset_prices_predict[i][:,1], label="", color=:blue, alpha=0.1)
end
plt

# Compute the acumulated returns of a agent that always invests in the asset with the highest predicted mean return
returns = Vector{Vector{Float64}}(undef, number_scenarios)
for scen = 1:number_scenarios
    returns[scen] = Vector{Float64}(undef, num_t)
    for t=1:num_t
        idx_observed = 1:t
        x_observed = x[idx_observed]
        y_observed = factors[idx_observed, scen, :]
        idx_predict = t+1:t+1
        x_predict = x[idx_predict]
        y_predict, asset_prices_predict = calculate_posterior_returns(factors_naive, x_observed, y_observed, x_predict, A)
        returns[scen][t] = maximum(mean.(asset_prices_predict))
    end
end

plt = plot(accumulate(+, returns[1]), label="")
for i in 2:number_scenarios
    plot!(plt, accumulate(+, returns[i]), label="")
end
plt