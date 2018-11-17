##############################
## by Qin Yu, Nov 2018
## using Julia 1.0.1
##############################

# If you have not installed these packages:
# using Pkg # Julia 1.0.1 only
# Pkg.add("GR")
# Pkg.add("Plots")
# Pkg.add("PyPlot")
# Pkg.add("Distributions")

# I will not include Pkg.add info in future .jl files,
# add it if you can't be "using" any.

############################## Starting here:
using LinearAlgebra
using Plots
using Random
using Distributions
# Load my own package:
push!(LOAD_PATH, ".")
using SuperLearn

############################## Change Basis, Run:
# Core.eval(SuperLearn, :(POLY_OR_SINE = "poly"))  # Change to Polynomial Basis (default)
# Core.eval(SuperLearn, :(POLY_OR_SINE = "sine"))  # Change to Sine Basis


############################## Another Simple Dataset:
# random function 𝑔_σ(𝑥) := sin²(2π𝑥) + ϵ, where random var ϵ ~ 𝐍(0, σ²)
# sample size n = 30, iid random variable 𝑋ᵢ ~ 𝐔(0, 1)
# srand(123)  # Julia 0.6.4
Random.seed!(777)  # Julia 1.0.1

# Define a distribution that we are sampling from, and sample:
𝑥ᵢ = Uniform(0, 1)
𝒙 = rand(𝑥ᵢ, 30)
sort!(𝒙)

# Plot the function sin²(2πx) in the range 0 ≤ x ≤ 1
f(x) = (sin(2 * π * x))^2
plot(f, 0, 1, lab="sin²(2 pi x)")

# with the points of the above data set superimposed:
g(σ, x) = f(x) + rand(Normal(0, σ), 1)[1]  # add noise when we sample
g₀₀₇(x) = g(0.07, x)
𝒚 = vcat(g₀₀₇.(𝒙)...)
scatter!(𝒙, 𝒚, lab="S")

savefig("2.1.pdf")

############################## Train 20 regressions:
# I am using trained_regression_line() from my own package,
# which is extracted from 1.Linear_Regression_Basis_Function.jl
regression_curves = [x -> trained_regression_line(x, 𝒙, 𝒚, i) for i in 1:20]

# Plot selected learned linear regression curves (every third):
for i = 1:3:size(regression_curves, 1)
    display(plot(regression_curves[i], 0, 1, lab="k = $i"))
    display(scatter!(𝒙, 𝒚, lab="S"))
    savefig("2.2-$i.pdf")
end


############################## Training Errors for Different Sizes of Basis:
te_k(k) = training_error_k_dim_basis(𝒙, 𝒚, k)
te_k_1to20 = te_k.(1:20)
log_te_k_1to20 = log.(te_k_1to20)
#plot(1:20, te_k_1to20)
plot(1:20, log_te_k_1to20, xlabel="k", xticks=0:20, ylabel="log training MSE", lab="log(te(k, S))")

############################## Testing Error with Testing Set of Size 1000:
𝒙_test = rand(𝑥ᵢ, 1000)
sort!(𝒙_test)
𝒚_test = vcat(g₀₀₇.(𝒙_test)...)
scatter(𝒙_test, 𝒚_test)

tse_k(k) = test_error_k_dim_basis(𝒙_test, 𝒚_test, 𝒙, 𝒚, k)
tse_k_1to20 = tse_k.(1:20)
log_tse_k_1to20 = log.(tse_k_1to20)
#plot!(1:20, tse_k_1to20)
plot(1:20, log_te_k_1to20, xlabel="k", xticks=0:20, ylabel="log MSE", lab="log(te(k, S))")
plot!(1:20, log_tse_k_1to20, lab="log(tse(k, S, T))")
# tse_k(k) = test_error_k_dim_basis(𝒙, 𝒚, 𝒙, 𝒚, k)  # Testing if this func's correctness
savefig("2.3.pdf")

############################## Obtain Previous Result 100 Times and Get Log-Average:
sum_all_100_te = fill(0, 20)
sum_all_100_tse = fill(0, 20)
for n = 1:100
    𝒙 = rand(𝑥ᵢ, 30)  # increasing the dimension of training set REDUCEs the error
    sort!(𝒙)
    𝒚 = vcat(g₀₀₇.(𝒙)...)
    te_k(k) = training_error_k_dim_basis(𝒙, 𝒚, k)
    te_k_1to20 = te_k.(1:20)
    global sum_all_100_te += te_k.(1:20)

    𝒙_test = rand(𝑥ᵢ, 1000)
    sort!(𝒙_test)
    𝒚_test = vcat(g₀₀₇.(𝒙_test)...)
    tse_k(k) = test_error_k_dim_basis(𝒙_test, 𝒚_test, 𝒙, 𝒚, k)
    tse_k_1to20 = tse_k.(1:20)
    global sum_all_100_tse += tse_k.(1:20)
end

avg_all_100_te = sum_all_100_te / 100
avg_all_100_tse = sum_all_100_tse / 100
log_all_100_te = log.(avg_all_100_te)
log_all_100_tse = log.(avg_all_100_tse)
plot(1:20, log_all_100_te, xlabel="k", xticks=0:20, ylabel="log average MSE", lab="log(avg(te(k, S)))")
plot!(1:20, log_all_100_tse, lab="log(avg(tse(k, S, T)))")  # run this section again, the larger k tail varies
savefig("2.4.pdf")


############################## To Change Polynomial Basis to Sine Basis, Run:
Core.eval(SuperLearn, :(POLY_OR_SINE = "sine"))
# Changed the basis, run everything again.
