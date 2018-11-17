##############################
## by Qin Yu, Nov 2018
## using Julia 1.0.1
##############################

############################## Starting here:
using LinearAlgebra
using Statistics
using Random
using Plots
using MAT  # Windows only
using JLD  # macOS Friendly
push!(LOAD_PATH, ".")
using SuperLearn
gr()

############################## The Boston Housing Dataset:
# Detailed Description from Toronto University:
### http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# MAT File from Prof. Mark Herbster, University College London:
### http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston

# This dataset contains information collected by the U.S Census Service
#   concerning housing in the area of Boston Mass. It was obtained
#   from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and
#   has been used extensively throughout the literature to benchmark algorithms.
# However, these comparisons were primarily done outside of Delve
#   and are thus somewhat suspect.
# The dataset is small in size with only 506 cases.
# The data was originally published
#   by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air',
#   J. Environ. Economics & Management, vol.5, 81-102, 1978.

#------------------------------
# The Boston Housing Dataset:
# There are 14 attributes in each case of the dataset. They are:
# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's
#------------------------------


############################## Processing From :
#----- (for my Windows 10)
# Import Boston:
matdata = matread("boston.mat")["boston"]
# If MAT package is not available for your version of Julia,
# use JLD to prepare data on another machine:
save("./boston.jld", "boston", matdata)

#----- (for my macOS)
# use JLD to load the transfored data:
matdata = load("boston.jld")["boston"]
# and have a look at all data of y:
plot(matdata[:, 14])


############################## Naive Regression (i.e. Linear Regression with 0 Attribute):
# Generating test/training sets:
nrow, ncol = size(matdata)
nrow_test  = div(nrow, 3)
nrow_train = nrow - nrow_test
ones_test  = ones(nrow_test)
ones_train = ones(nrow_train)

# 20 runs of randomly splitted datasets:
plot(matdata[:, 14], lab="boston[:,14]")
sum_all_20_te = sum_all_20_tse = 0
for i = 1:20
    # Obtain randomly splitted 𝒚_test/training:
    permuted_matdata = matdata[randperm(nrow),:]
    𝒚_train = permuted_matdata[nrow_test+1:nrow, 14]
    𝒚_test  = permuted_matdata[1:nrow_test, 14]

    # Plot each run:
    trl(x_test) = trained_regression_line(x_test, ones_train, 𝒚_train, 1)
    plot!(trl, 1, nrow, lab="$i th run")

    # Obtain MSEs:
    global sum_all_20_te += training_error_k_dim_basis(ones_train, 𝒚_train, 1)
    global sum_all_20_tse += test_error_k_dim_basis(ones_test, 𝒚_test, ones_train, 𝒚_train, 1)
end
current()
savefig("3.1.pdf")
# Obtain average MSEs for testing/training over 20 runs:
avg_all_20_te  = sum_all_20_te / 20
avg_all_20_tse = sum_all_20_tse / 20


############################## Linear Regression with 1 Attribute:
# Using SuperLearn without basis function (but with integrated bias/intercept term, [ ,1]):
Core.eval(SuperLearn, :(TRANS_BASIS = false))

# the 𝑖th element is for the 𝑖th 𝑥, and the last entry saved for next section: all-attributes-regression
sum_all_20_te = sum_all_20_tse = avg_all_20_te = avg_all_20_tse = zeros(14)
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

        sum_all_20_te[i]  += training_error_k_dim_basis(𝒙_train, 𝒚_train, nothing)
        sum_all_20_tse[i] += test_error_k_dim_basis(𝒙_test, 𝒚_test, 𝒙_train, 𝒚_train, nothing)
        j == 1 && savefig("3.2-$i.pdf")
    end
end
avg_all_20_te  = sum_all_20_te / 20
avg_all_20_tse = sum_all_20_tse / 20


############################## Linear Regression with All Attributes
plot(matdata[:, 14])
X_test  = matdata[1:nrow_test, 1:13]
X_train = matdata[nrow_test+1:nrow, 1:13]
𝒚_test  = matdata[1:nrow_test, 14]
𝒚_train = matdata[nrow_test+1:nrow, 14]
trl(X_test) = trained_regression_line_q4(X_test, X_train, 𝒚_train, nothing)
scatter!(nrow_test+1:nrow, trl(X_train))

# Have a look at our training result:
sorted_matdata = sort_matrix_by_jth_col(matdata, 14)
plot(sorted_matdata[:, 14])
plot!(1:nrow, trl(sorted_matdata[:, 1:13]))


sum_all_20_te[14] = sum_all_20_tse[14] = 0
sorted_matdata = sort_matrix_by_jth_col(matdata, 14)
plot(sorted_matdata[:, 14], lab="real y value")
for j = 1:20
    permuted_matdata = matdata[randperm(nrow),:]
    X_test  = permuted_matdata[1:nrow_test, 1:13]
    X_train = permuted_matdata[nrow_test+1:nrow, 1:13]
    𝒚_test  = permuted_matdata[1:nrow_test, 14]
    𝒚_train = permuted_matdata[nrow_test+1:nrow, 14]

    trl(X_test) = trained_regression_line_q4(X_test, X_train, 𝒚_train, nothing)
    display(plot!(1:nrow, trl(sorted_matdata[:, 1:13]), lab="$j th run"))  # current() outside loop, display() inside!!

    sum_all_20_te[14]  += training_error_k_dim_basis(X_train, 𝒚_train, nothing)
    sum_all_20_tse[14] += test_error_k_dim_basis(X_test, 𝒚_test, X_train, 𝒚_train, nothing)
end
savefig("3.3.pdf")
avg_all_20_te[14]  = sum_all_20_te[14] / 20
avg_all_20_tse[14] = sum_all_20_tse[14] / 20

############################## Comparing the Result (More Detailed Comparisons in 4.jl)
plot(avg_all_20_te, xlabel="attributes, 14th is all-attribute", xticks=0:14, ylabel="MSE", lab="avg training MSE")
plot!(avg_all_20_tse, lab="avg testing MSE")
savefig("3.4.pdf")
