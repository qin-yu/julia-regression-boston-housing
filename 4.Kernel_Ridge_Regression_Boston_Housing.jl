##############################
## by Qin Yu, UCL, Nov 2018
## using Julia 1.0.1
##############################

############################## Starting here:
using LinearAlgebra
using Statistics
using Random
using Plots
using MAT  # Windows only
using JLD  # macOS Friendly
using CSV
using Printf
using DataFrames
push!(LOAD_PATH, ".")
using SuperLearn
gr()


############################## The Boston Housing Dataset:
## Detail see 3.Linear_Regression_Boston_Housing.jl


############################## Import Boston:
#----- (for my Windows 10)
# Import Boston:
matdata = matread("boston.mat")["boston"]
# If MAT package is not available for your version of Julia,
# use JLD to prepare data on another machine:
# save("./boston.jld", "boston", matdata)

#----- (for my macOS)
# use JLD to load the transfored data:
matdata = load("boston.jld")["boston"]
# and have a look at all data of y:
# plot(matdata[:, 14])


############################## 5-fold Cross-validation to Find Best Parameter Pair
# Define a pool of parameters, we want to find the best pair:
𝛄 = [2.0^i for i in -40:-26]  # regularisation parameter γ, and
𝛔 = [2.0^i for i in collect(7:.5:13)]  # variance parameter σ for Gaussian kernel

# Prepare 5-folds data:
nrow, ncol = size(matdata)
nrow_test  = div(nrow, 3)
nrow_train = nrow - nrow_test
Random.seed!(1111)  # this number is again pure magic
permuted_matdata = matdata[randperm(nrow),:]
permuted_matdata_train = permuted_matdata[1:nrow_train,:]
permuted_matdata_test = permuted_matdata[nrow_train+1:end,:]

nrow_5fold  = div(nrow_train, 5)
𝑙 = nrow_train - nrow_5fold
𝑰 = Matrix{Float64}(I, 𝑙, 𝑙)  # 𝑰 = Id(l by l)

𝐾(𝒙ᵢ, 𝒙ⱼ, σ) = exp(-(norm(𝒙ᵢ - 𝒙ⱼ)^2)/(2 * σ^2))  # 𝐾(𝒙ᵢ, 𝒙ⱼ) = 𝑒^{∥𝒙ᵢ-𝒙ⱼ∥²/2σ²}
function fill_𝑲(𝑲, 𝑿, σ)
    nrow_𝑲, ncol_𝑲 = size(𝑲)
    for i in 1:nrow_𝑲
        for j in 1:ncol_𝑲
            𝑲[i, j] = 𝐾(𝑿[i,:], 𝑿[j,:], σ)  # 𝐾ᵢⱼ = 𝐾(𝒙ᵢ, 𝒙ⱼ)
        end
    end
end
get_𝜶(𝑲, γ, 𝑰, 𝒚) = inv(𝑲 + (γ * 𝑙 * 𝑰)) * 𝒚  # 𝛂* = (𝑲 + γ𝑙𝑰)⁻¹⋅𝒚
function get_ŷ(𝜶, x, σ, γ, 𝑿_train)  # 𝒙ₐ denoted by x
    𝐾ₐ(𝒙ᵢ, σ) = 𝐾(𝒙ᵢ, x', σ)
    𝑿_train_vectorised = [𝑿_train[i,:]' for i = 1:size(𝑿_train, 1)]
    ŷ = 𝜶' * 𝐾ₐ.(𝑿_train_vectorised, σ)  # ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ
end

SSE = MSE = zeros(size(𝛔,1), size(𝛄,1))
# DO NOT RUN THE TRI-LOOP!!! because otherwise you need to wait.
# If you don't want to wait, call these 2 lines to load my data:
# SSE = load("SSE.jld")["SSE"]
# MSE = load("MSE.jld")["MSE"]
# If you run the 2 lines above, don't /5 later.
for σ in 𝛔
    σ_index = findfirst(𝛔 .== σ)
    for γ in 𝛄
        γ_index = findfirst(𝛄 .== γ)
        for i = 0:4
            # get 5-fold data:
            𝑿_test = permuted_matdata_train[(i*nrow_5fold + 1):((i+1) * nrow_5fold), 1:13]
            𝒚_test = permuted_matdata_train[(i*nrow_5fold + 1):((i+1) * nrow_5fold), 14]

            𝑿_train_1 = permuted_matdata_train[1:(i*nrow_5fold), 1:13]
            𝑿_train_2 = permuted_matdata_train[((1+i)*nrow_5fold+1):end, 1:13]
            𝑿_train = vcat(𝑿_train_1, 𝑿_train_2)

            𝒚_train_1 = permuted_matdata_train[1:(i*nrow_5fold), 14]
            𝒚_train_2 = permuted_matdata_train[((1+i)*nrow_5fold+1):end, 14]
            𝒚_train = vcat(𝒚_train_1, 𝒚_train_2)

            # compute 𝒚̂:
            𝑲 = zeros(𝑙, 𝑙)
            fill_𝑲(𝑲, 𝑿_train, σ)  # compute kernel matrix, # 𝐾ᵢⱼ = 𝐾(𝒙ᵢ, 𝒙ⱼ)

            𝜶 = get_𝜶(𝑲, γ, 𝑰, 𝒚_train)  # compute 𝛂*, 𝛂* = (𝑲 + γ𝑰)⁻¹⋅𝒚

            𝑿_test_vectorised = [𝑿_test[i,:] for i = 1:size(𝑿_test, 1)]
            get_ŷ_(x) = get_ŷ(𝜶, x, σ, γ, 𝑿_train)
            ŷ = get_ŷ_.(𝑿_test_vectorised)  # compute ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ

            # have a look:
            display(plot(ŷ))
            display(plot!(𝒚_test))

            # compute testing error, MSE:
            sse = sum((𝒚_test - ŷ).^2)  # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)²
            mse = sse/nrow_5fold  # MSE = SSE/N
            SSE[σ_index, γ_index] += sse
            MSE[σ_index, γ_index] += mse
        end
    end
end
SSE /= 5
MSE /= 5

# To save it so that I don't need to run again:
save("./data/SSE.jld", "SSE", SSE)
save("./data/MSE.jld", "MSE", MSE)

min_mse, σ_γ_indices = findmin(MSE)
σ_index, γ_index = σ_γ_indices[1], σ_γ_indices[2]

# Plot heatmap, take log so that the colourscale makes more sense:
heatmap(log2.(𝛄), log2.(𝛔), xticks=-40:-26, xlabel="gamma",
                            yticks=7:.5:13, ylabel="sigma", log.(log.(SSE)))
savefig("4.1.pdf")


############################## !! Comparing Everything !!
# 1 for naive regression
# 2-14 for 1-attribute linear regression
# 15 for all-attribute linear regression
jl4_𝐄 = zeros(16, 2)
jl4_𝛔 = zeros(16, 2)

# You can skip to last part by loading these:
# jl4_𝐄 = load("jl4_E.jld")["E"]
# jl4_𝛔 = load("jl4_s.jld")["s"]
############################## ADDING: Kernel Ridge Regression
# Run on whole training and testing sets:
σ = 𝛔[σ_index]
γ = 𝛄[γ_index]

𝑿_test  = permuted_matdata_test[:, 1:13]
𝑿_train = permuted_matdata_train[:, 1:13]
𝒚_test  = permuted_matdata_test[:, 14]
𝒚_train = permuted_matdata_train[:, 14]

𝑙 = nrow_train
𝑰 = Matrix{Float64}(I, 𝑙, 𝑙)  # 𝑰 = Id(l x l)

𝑲 = zeros(𝑙, 𝑙)
fill_𝑲(𝑲, 𝑿_train, σ)  # compute kernel matrix, # 𝐾ᵢⱼ = 𝐾(𝒙ᵢ, 𝒙ⱼ)

𝜶 = get_𝜶(𝑲, γ, 𝑰, 𝒚_train)  # compute 𝛂*, 𝛂* = (𝑲 + γ𝑰)⁻¹⋅𝒚

𝑿_train_vectorised = [𝑿_train[i,:] for i = 1:size(𝑿_train, 1)]
get_ŷ_(x) = get_ŷ(𝜶, x, σ, γ, 𝑿_train)
ŷ_train = get_ŷ_.(𝑿_train_vectorised)  # compute ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ

𝑿_test_vectorised = [𝑿_test[i,:] for i = 1:size(𝑿_test, 1)]
get_ŷ_(x) = get_ŷ(𝜶, x, σ, γ, 𝑿_train)
ŷ_test = get_ŷ_.(𝑿_test_vectorised)  # compute ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ

sse_train = sum((𝒚_train - ŷ_train).^2)  # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)²
mse_train = sse_train/nrow_train  # MSE = SSE/N

sse_test = sum((𝒚_test - ŷ_test).^2)  # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)²
mse_test = sse_test/nrow_test  # MSE = SSE/N

sse20_test  = zeros(20)
sse20_train = zeros(20)
for i = 1:20
    permuted_matdata = matdata[randperm(nrow),:]
    permuted_matdata_train = permuted_matdata[1:nrow_train,:]
    permuted_matdata_test = permuted_matdata[nrow_train+1:end,:]

    σ = 𝛔[σ_index]
    γ = 𝛄[γ_index]

    𝑿_test  = permuted_matdata_test[:, 1:13]
    𝑿_train = permuted_matdata_train[:, 1:13]
    𝒚_test  = permuted_matdata_test[:, 14]
    𝒚_train = permuted_matdata_train[:, 14]

    𝑙 = nrow_train
    𝑰 = Matrix{Float64}(I, 𝑙, 𝑙)  # 𝑰 = Id(l x l)

    𝑲 = zeros(𝑙, 𝑙)
    fill_𝑲(𝑲, 𝑿_train, σ)  # compute kernel matrix, # 𝐾ᵢⱼ = 𝐾(𝒙ᵢ, 𝒙ⱼ)

    𝜶 = get_𝜶(𝑲, γ, 𝑰, 𝒚_train)  # compute 𝛂*, 𝛂* = (𝑲 + γ𝑰)⁻¹⋅𝒚

    𝑿_train_vectorised = [𝑿_train[i,:] for i = 1:size(𝑿_train, 1)]
    get_ŷ_(x) = get_ŷ(𝜶, x, σ, γ, 𝑿_train)
    ŷ_train = get_ŷ_.(𝑿_train_vectorised)  # compute ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ

    𝑿_test_vectorised = [𝑿_test[i,:] for i = 1:size(𝑿_test, 1)]
    get_ŷ_(x) = get_ŷ(𝜶, x, σ, γ, 𝑿_train)
    ŷ_test = get_ŷ_.(𝑿_test_vectorised)  # compute ŷ = 𝜶ᵀ⋅(𝐾₁ₐ, 𝐾₂ₐ, ..., 𝐾ₗₐ)ᵀ

    sse20_train[i] = sum((𝒚_train - ŷ_train).^2)  # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)²
    #mse_train = sse_train/nrow_train  # MSE = SSE/N

    sse20_test[i] = sum((𝒚_test - ŷ_test).^2)  # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)²
    #mse_test = sse_test/nrow_test  # MSE = SSE/N
end

mse20_train = sse20_train/nrow_train
mse20_test  = sse20_test/nrow_test
plot(mse20_train, lab="training error")
plot!(mse20_test, lab="testing error")
savefig("4.2.pdf")

# x₁, ..., xₙ are n independent obeservations from a population
# that has mean μ and variance σ, then the variance of Σxᵢ is nσ²
# and the variance of x̄ = Σxᵢ/n is σ²/n
jl4_𝐄[16, 1] = 𝐄mse_train = mean(mse20_train)
jl4_𝐄[16, 2] = 𝐄mse_test  = mean(mse20_test)

jl4_𝛔[16, 1] = se_train = std(mse20_train)
jl4_𝛔[16, 2] = se_test  = std(mse20_test)


############################## ADDING: Naive Regression
# 20 runs of randomly splitted datasets:
ones_test  = ones(nrow_test)
ones_train = ones(nrow_train)
plot(matdata[:, 14])
sum_all_20_te  = zeros(20)  # no longer sum
sum_all_20_tse = zeros(20)
for i = 1:20
    # Obtain randomly splitted 𝒚_test/training:
    permuted_matdata = matdata[randperm(nrow),:]
    𝒚_train = permuted_matdata[nrow_test+1:nrow, 14]
    𝒚_test  = permuted_matdata[1:nrow_test, 14]

    # Plot each run:
    trl(x_test) = trained_regression_line(x_test, ones_train, 𝒚_train, 1)
    plot!(trl, 1, nrow)
    #println("done")

    # Obtain MSEs:
    sum_all_20_te[i]  = training_error_k_dim_basis(ones_train, 𝒚_train, 1)
    sum_all_20_tse[i] = test_error_k_dim_basis(ones_test, 𝒚_test, ones_train, 𝒚_train, 1)
end

# Show the result of plot loop by current():
current()

jl4_𝐄[1, 1] = 𝐄mse_naive_train = mean(sum_all_20_te)
jl4_𝐄[1, 2] = 𝐄mse_naive_test  = mean(sum_all_20_tse)

jl4_𝛔[1, 1] = se_naive_train = std(sum_all_20_te)
jl4_𝛔[1, 2] = se_naive_test  = std(sum_all_20_tse)


############################## ADDING: Linear Regression (attribute i)
# Using SuperLearn without basis function (but with integrated [ ,1]):
Core.eval(SuperLearn, :(TRANS_BASIS = false))

# the 𝑖th element is for the 𝑖th 𝑥:
sum_all_20_te  = zeros(14, 20)  # This is called sum for my historical reason
sum_all_20_tse = zeros(14, 20)
for j = 1:20
    permuted_matdata = matdata[randperm(nrow),:]
    for i = 1:13
        𝒙_test  = permuted_matdata[1:nrow_test, i]
        𝒙_train = permuted_matdata[nrow_test+1:nrow, i]
        𝒚_test  = permuted_matdata[1:nrow_test, 14]
        𝒚_train = permuted_matdata[nrow_test+1:nrow, 14]
        scatter(𝒙_train, 𝒚_train)
        trl(x_test) = trained_regression_line(x_test, 𝒙_train, 𝒚_train, nothing)
        display(plot!(trl, minimum(𝒙_train), maximum(𝒙_train)))

        sum_all_20_te[i, j]  = training_error_k_dim_basis(𝒙_train, 𝒚_train, nothing)
        sum_all_20_tse[i, j] = test_error_k_dim_basis(𝒙_test, 𝒚_test, 𝒙_train, 𝒚_train, nothing)
    end
end

jl4_𝐄[2:14, 1] = 𝐄mse_xi_train = [mean(sum_all_20_te[i,:])  for i in 1:13]
jl4_𝐄[2:14, 2] = 𝐄mse_xi_test  = [mean(sum_all_20_tse[i,:]) for i in 1:13]

jl4_𝛔[2:14, 1] = se_xi_train = [std(sum_all_20_te[i,:])  for i in 1:13]
jl4_𝛔[2:14, 2] = se_xi_test  = [std(sum_all_20_tse[i,:]) for i in 1:13]


############################## ADDING: Linear Regression (all attribute)
plot(matdata[:, 14])
X_test  = matdata[1:nrow_test, 1:13]
X_train = matdata[nrow_test+1:nrow, 1:13]
𝒚_test  = matdata[1:nrow_test, 14]
𝒚_train = matdata[nrow_test+1:nrow, 14]
trl(X_test) = trained_regression_line_M(X_test, X_train, 𝒚_train, nothing)
scatter!(nrow_test+1:nrow, trl(X_train))  # current() outside loop, display() inside!!

sorted_matdata = sort_matrix_by_jth_col(matdata, 14)
for j = 1:20
    permuted_matdata = matdata[randperm(nrow),:]
    X_test  = permuted_matdata[1:nrow_test, 1:13]
    X_train = permuted_matdata[nrow_test+1:nrow, 1:13]
    𝒚_test  = permuted_matdata[1:nrow_test, 14]
    𝒚_train = permuted_matdata[nrow_test+1:nrow, 14]

    sum_all_20_te[14, j]  = training_error_k_dim_basis(X_train, 𝒚_train, nothing)
    sum_all_20_tse[14, j] = test_error_k_dim_basis(X_test, 𝒚_test, X_train, 𝒚_train, nothing)
end

jl4_𝐄[15, 1] = 𝐄mse_xall_train = mean(sum_all_20_te[14,:])
jl4_𝐄[15, 2] = 𝐄mse_xall_test  = mean(sum_all_20_tse[14,:])

jl4_𝛔[15, 1] = se_xall_train = std(sum_all_20_te[14,:])
jl4_𝛔[15, 2] = se_xall_test  = std(sum_all_20_tse[14,:])

jl4_𝐄
jl4_𝛔
save("./data/jl4_E.jld", "E", jl4_𝐄)
save("./data/jl4_s.jld", "s", jl4_𝛔)

# Print Table:
# jl4_𝐄 = load("jl4_E.jld")["E"]
# jl4_𝛔 = load("jl4_s.jld")["s"]
jl4_𝐄_𝛔 = hcat(jl4_𝐄[:,1], jl4_𝛔[:,1], jl4_𝐄[:,2], jl4_𝛔[:,2])
jl4_𝐄_𝛔_rd = round.(jl4_𝐄_𝛔, digits=4)
jl4_𝐄_𝛔_str = (x -> @sprintf("%.2f", x)).(jl4_𝐄_𝛔_rd)
jl4_𝐄_str = (x -> @sprintf("%.2f", x)).(jl4_𝐄)
jl4_𝛔_str = (x -> @sprintf("%.2f", x)).(jl4_𝛔)
jl4 = jl4_𝐄_str .* " ± " .* jl4_𝛔_str

q5_DataFrame = convert(DataFrame, jl4)
names!(q5_DataFrame, [:E, :sigma])
q5_DataFrame[:Regression] = vcat(["Naive"], ["x$i" for i in 1:13], ["x"], ["KRR"])
q5_DataFrame = q5_DataFrame[[:Regression, :E, :sigma]]

print(q5_DataFrame)
io = open("./data/final_compare.txt", "w")
print(io, q5_DataFrame)
close(io)

# 16×3 DataFrame
# │ Row │ Regression │ E            │ sigma         │
# │     │ String     │ String       │ String        │
# ├─────┼────────────┼──────────────┼───────────────┤
# │ 1   │ Naive      │ 82.03 ± 6.00 │ 89.53 ± 12.30 │
# │ 2   │ x1         │ 72.15 ± 4.32 │ 71.45 ± 8.54  │
# │ 3   │ x2         │ 74.20 ± 4.12 │ 72.22 ± 8.27  │
# │ 4   │ x3         │ 65.28 ± 3.77 │ 63.75 ± 7.48  │
# │ 5   │ x4         │ 82.31 ± 4.31 │ 81.25 ± 8.66  │
# │ 6   │ x5         │ 69.85 ± 3.73 │ 67.59 ± 7.25  │
# │ 7   │ x6         │ 43.50 ± 4.35 │ 44.30 ± 9.29  │
# │ 8   │ x7         │ 73.26 ± 3.83 │ 71.03 ± 7.53  │
# │ 9   │ x8         │ 80.05 ± 4.53 │ 77.61 ± 9.00  │
# │ 10  │ x9         │ 72.69 ± 4.61 │ 71.30 ± 9.20  │
# │ 11  │ x10        │ 66.64 ± 4.22 │ 64.70 ± 8.38  │
# │ 12  │ x11        │ 62.96 ± 4.27 │ 62.41 ± 8.58  │
# │ 13  │ x12        │ 75.28 ± 4.30 │ 74.74 ± 8.49  │
# │ 14  │ x13        │ 38.70 ± 2.03 │ 38.32 ± 4.03  │
# │ 15  │ x          │ 22.03 ± 1.57 │ 23.13 ± 3.53  │
# │ 16  │ KRR        │ 8.13 ± 0.76  │ 12.90 ± 2.16  │
