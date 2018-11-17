module SuperLearn

##############################
## by Qin Yu, Nov 2018
## using Julia 1.0.1
## for Supervised Learning
##############################

using LinearAlgebra # Julia 1.0.1 only
using Plots
using Printf
using Statistics

export TRANS_BASIS,
       POLY_OR_SINE,
       sort_matrix_by_jth_col,
       trained_regression_line,
       trained_regression_line_M,
       test_error_k_dim_basis,
       training_error_k_dim_basis,
       get_se

POLY_OR_SINE = "poly"
TRANS_BASIS = true

# Formulae - Find 𝝎:
function phik(x, k)  # function phik(x, k; basis="poly")  # don't know how to implement this efficiently
    if POLY_OR_SINE == "poly"
        [xi^(k-1) for xi in x]  # ϕₖ(𝒙) = ..., for vector 𝒙, all inputs
    elseif POLY_OR_SINE == "sine"
        [sin(k * π * xi) for xi in x]  # ϕₖ(𝒙) = ..., for vector 𝒙, all inputs
    end
end

phi1tok(x, k) = [phik(x, i) for i in 1:k]  # Φ(𝒙) = 𝒙 ⋅ 𝝋 = 𝒙 ⋅ (ϕ₁(), ϕ₂(), ..., ϕₖ())

function transformed_x_kk(x, k)  # Φ(𝒙) as a matrix
    if TRANS_BASIS
        hcat(phi1tok(x, k)...)
    else
        size(x) == () ? [x, 1] : hcat(x, ones(size(x)[1]))
    end
end

𝒘(x, y, k) = transformed_x_kk(x, k) \ y  # 𝝎 = Φ\𝒚

# Formulae - Equation of Fitted Regression Line:
𝒘Φ_M(X_test, X_train, 𝒚_train, nothing) = transformed_x_kk(X_test, nothing) * 𝒘(X_train, 𝒚_train, nothing)
𝒘Φ(x_test, x_train, y, k) = dot(transformed_x_kk(x_test, k), 𝒘(x_train, y, k))  # ̂𝑦 = Φ(𝑥) ⋅ 𝝎
trained_regression_line = 𝒘Φ
trained_regression_line_M = 𝒘Φ_M

# MSE for Testing:
function test_error_k_dim_basis(𝒙_test, 𝒚_test, 𝒙, 𝒚, k)
    # SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)² = 𝚺ᵢ(𝑦ᵢ - Φ(𝑥ᵢ) ⋅ 𝝎)²
    # MSE = SSE/N, where N = number_of_rows(input_data_set), here is S
    sse = sum((𝒚_test - transformed_x_kk(𝒙_test, k) * 𝒘(𝒙, 𝒚, k)).^2)
    mse = sse / first(size(𝒚_test))
    return mse
end

# MSE for Training:
training_error_k_dim_basis(𝒙, 𝒚, k) = test_error_k_dim_basis(𝒙, 𝒚, 𝒙, 𝒚, k)

# Sort a Matrix by the jth Column:
function sort_matrix_by_jth_col(A, j)
    disassembled_A = [A[i,:] for i in 1:size(A, 1)]
    sort!(disassembled_A, by = x -> x[j])
    vcat(disassembled_A'...)
end

# My function of get standard_error from observations:
function get_se(observations)
    n = size(observations, 1)
    standard_deviation = std(observations)
    standard_error = standard_deviation / sqrt(n)
    return standard_error
end

end
