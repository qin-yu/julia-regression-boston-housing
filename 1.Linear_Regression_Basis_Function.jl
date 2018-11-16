##############################
## by Qin Yu, Nov 2018
## using Julia 1.0.1
##############################

############################## Starting here:
using LinearAlgebra # Julia 1.0.1 only
using Plots
using Printf

############################## Linear Regression with Polynomial Basis on Simple Dataset:
# Prepare Data
# Let's try to see the effect of polynomial basis,
# on this extremely simple dataset, S:
S = [(1,3),(2,2),(3,0),(4,5)]
plot(S, line=:scatter, lab="data set")

x = [x[1] for x in S]
y = [y[2] for y in S]

# Formulae - Find 𝝎:
phik(x, k) = [xi^k for xi in x]  # ϕₖ(𝒙) = ..., for vector 𝒙, all inputs
phi1tok(x, k) = [phik(x, i) for i in 0:k-1]  # Φ(𝒙) = 𝒙 ⋅ 𝝋 = 𝒙 ⋅ (ϕ₁(), ϕ₂(), ..., ϕₖ())
transformed_x_kk(x, k) = hcat(phi1tok(x, k)...)  # Φ(𝒙) as a matrix
w_k(x, y, k) = transformed_x_kk(x, k) \ y  # 𝝎 = Φ\𝒚

# Formulae - Equation of Fitted Regression Line:
phi_k(x_test, x_train, y, k) = dot(transformed_x_kk(x_test, k), w_k(x_train, y, k))  # ̂𝑦 = Φ(𝑥) ⋅ 𝝎

plot(S, line=:scatter, lab="data set", legend=:bottomright)
W = zeros(4, 4)
for i = 1:4
    𝒘Φ(x_test) = phi_k(x_test, x, y, i)
    display(plot!(𝒘Φ, 0, 4, lab="k = $i"))
    W[i,1:i] = w_k(x, y, i)
end
@printf "k = 1, f(x) = %.2f\n" W[1,1:1]...
@printf "k = 2, f(x) = %.2f + %.2f𝑥\n" W[2,1:2]...
@printf "k = 3, f(x) = %.2f + %.2f𝑥 + %.2f𝑥²\n" W[3,1:3]...
@printf "k = 4, f(x) = %.2f + %.2f𝑥 + %.2f𝑥² + %.2f𝑥³\n" W[4,1:4]...
savefig("1.1.pdf")


############################## Training Error
# SSE = 𝚺ᵢ(𝑦ᵢ - ̂𝑦ᵢ)² = 𝚺ᵢ(𝑦ᵢ - Φ(𝑥ᵢ) ⋅ 𝝎)²
# MSE = SSE/N, where N = number_of_rows(input_data_set), here is S
# (here I use N as the book ESLII uses, standing for m in qestions)
sse_k(x, y, k) = sum((y - transformed_x_kk(x, k) * w_k(x, y, k)).^2)
mse_k(x, y, k) = sse_k(x, y, k) / first(size(y))

MSE = [mse_k(x, y, i) for i = 1:4]
plot(MSE, xlabel="k", lab="MSE")
savefig("1.2.pdf")
