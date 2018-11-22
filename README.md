# Machine Learning Using Julia v1.0 on Boston Housing Dataset 
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/qin-yu/ml-julia-boston-housing/blob/master/LICENSE)
[![Julia v1.0.1](https://img.shields.io/badge/Julia-v1.0.1-brightgreen.svg)](https://julialang.org/blog/2018/08/one-point-zero)

- [Manifesto](#manifesto)
- [Overview of Examples](#overview-of-examples)
  - [Why I chose Julia](#why-i-chose-julia)
  - [How I Play with Julia](#how-i-play-with-julia)
  - [About The Boston Housing Dataset](#about-the-boston-housing-dataset)

## Manifesto
Julia 1.0 was released during JuliaCon in August 2018, two months before I started to look into machine learning. Dr David Barber, once in a lecture, expressed his faith in Julia's future success, so I decided to pick up Julia for a serious piece of coursework on supervised learning. To my surprise, my friends, including some of the most distinguished students, working on the same piece of coursework chose Python, MATLAB and R. However, Julia is fast(er?) and (more?) expressive, from my recent experience. More importantly, I had fun playing with it.

## Overview of Examples
[1] means the example can be found in the `.jl` file with names starting with "1"
- Linear regression with basis functions:
  - [1] polynomial basis & Super Simple Dataset (fit 4 points)
  - [2] polynomial basis & Simple Dataset (Sine + Gaussian Noise)
    - overfitting demo
  - [2] sine basis & Simple Dataset
    - overfitting demo
- [3] Linear regression with 0, 1, or more attributes:
  - naive regression (0 attribute linear regression)
    - in fact, this is just a fancy way of computing the expected value of 𝒚
  - linear regression with single attribute
  - linear regression with more than one attributes
- [4] Kernelised ridge regression (or Kernelised Tikhonov regularization):
  - using k-fold cross-validation to find best parameter pair:
    - regularisation parameter γ, and
    - variance parameter σ for Gaussian kernel
  - KRR with this pair of parameters
- [4] Comparing all methods in [3] and [4]

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

### About The Boston Housing Dataset
Boston housing is a classic dataset described in detail at [University of Toronto's Website](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html), and the data was originally published by Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

**Dataset Naming:**
The name for this dataset is simply boston. It has two prototasks: `nox`, in which the nitrous oxide level is to be predicted; and `price`, in which the median value of a home is to be predicted. **However, here, I am using everything to predict `price`.**

**Miscellaneous Details:**  
- Origin - The origin of the boston housing data is Natural.  
- Usage - This dataset may be used for Assessment.  
- Number of Cases - The dataset contains a total of 506 cases.  
- Order - The order of the cases is mysterious.  
- Variables - There are 14 attributes in each case of the dataset. They are:  
  `CRIM` - per capita crime rate by town  
  `ZN` - proportion of residential land zoned for lots over 25,000 sq.ft.  
  `INDUS` - proportion of non-retail business acres per town.  
  `CHAS` - Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
  `NOX` - nitric oxides concentration (parts per 10 million)  
  `RM` - average number of rooms per dwelling  
  `AGE` - proportion of owner-occupied units built prior to 1940  
  `DIS` - weighted distances to five Boston employment centres  
  `RAD` - index of accessibility to radial highways  
  `TAX` - full-value property-tax rate per $10,000  
  `PTRATIO` - pupil-teacher ratio by town  
  `B` - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
  `LSTAT` - % lower status of the population  
  `MEDV` - Median value of owner-occupied homes in $1000's  

"The boston housing data set as “.mat” ﬁle is located at [Prof. Mark Herbster's Website (UCL)](http://www.cs.ucl.ac.uk/staff/M.Herbster/boston) otherwise please go to URL above to retrieve in as a text ﬁle."
---- Prof. M. Herbster
