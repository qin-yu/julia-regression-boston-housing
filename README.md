# Machine Learning Using Julia v1.0 on Boston Housing Dataset

## Manifesto
Julia 1.0 was released during JuliaCon in August 2018, two months before I started to look into machine learning. Dr David Barber, once in a lecture, expressed his faith in Julia's future success, so I decided to pick up Julia for a serious piece of coursework on supervised learning. To my surprise, my friends, including some of the most distinguished students, working on the same piece of coursework chose Python, MATLAB and R. However, Julia is fast(er?) and (more?) expressive, from my recent experience. More importantly, I had fun playing with it.

## Overview of Examples
- Linear regression with basis functions:
  - polynomial basis
  - sine basis
  - overfitting demo
- Linear regression with 0, 1, or more attributes:
  - naive regression (0 attribute linear regression)
    - in fact, this is just a fancy way of computing the expected value of 𝒚
  - linear regression with single attribute
  - linear regression with more than one attributes
- Kernelised ridge regression (or Kernelised Tikhonov regularization):
  - using k-fold cross-validation to find best parameter pair:
    - regularisation parameter γ, and
    - variance parameter σ for Gaussian kernel
  - KRR with this pair of parameters,

### Why I chose Julia
- When it come to linear algebra, it is as powerful as MATLAB.
  - I can solve a linear system <img src="https://latex.codecogs.com/gif.latex?\inline&space;A\boldsymbol{x}&space;=&space;\boldsymbol{b}" title="A\boldsymbol{x} = \boldsymbol{b}" />, or find a least-square solution of a underdetermined system, by calling `A\b`.
  - To find <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{w}" title="\boldsymbol{w}" /> that minimizes <img src="https://latex.codecogs.com/gif.latex?\inline&space;||X\boldsymbol{w}&space;-&space;\boldsymbol{y}||^2" title="||X\boldsymbol{w} - \boldsymbol{y}||^2" />, one computes <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{w}&space;=&space;(X^TX)^{-1}X^T\boldsymbol{y}" title="\boldsymbol{w} = (X^TX)^{-1}X^T\boldsymbol{y}" />. In Julia, `X\y`.
- My code is comment, my code is concise, and Unicode is easy to code.
  - It feels great when it let me comment and name my variables in Unicode characters, especially with LaTeX-like abbreviations!
  - I can name my array of prediction <img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y}" title="\hat{y}" /> and each element <img src="https://latex.codecogs.com/gif.latex?\inline&space;y_i" title="y_i" />. No more "`alpha_hat`"-like variables, but <img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{\alpha}" title="\hat{\alpha}" />.
- It is interactive and it can have a workspace, just like Python and R, but it races with C!
- Officially, it is said to be **"Fast, General, Dynamic, Technical, Optionally typed, Composable"**

### How I Play with Julia
- I use [Juno](http://junolab.org), a beautiful IDE for Julia, built on Atom.
  - Here is the installation guide: http://docs.junolab.org/latest/man/installation.html
- If you really don't bother to install it and are familiar with Python and Jupyter Notebooks, which is highly likely because almost everyone I talked to used Python for this, just change the kernel to Julia and continue using Jupyter Notebooks.
  - Use [IJulia](https://github.com/JuliaLang/IJulia.jl) as your backend: https://github.com/JuliaLang/IJulia.jl
