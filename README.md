# Machine Learning Using Julia v1.0 on Boston Housing Dataset

## Manifesto
Julia 1.0 was released during JuliaCon in August 2018, two months before I started to look into machine learning. Dr David Barber, once in a lecture, expressed his faith in Julia's future success, so I decided to pick up Julia for a serious piece of coursework on supervised learning. To my surprise, my friends, including some of the most distinguished students, working on the same piece of coursework chose Python, MATLAB and R. However, Julia is fast(er?) and (more?) expressive, from my recent experience. More importantly, I had fun playing with it.

### Why I chose Julia
- When it come to linear algebra, it is as powerful as MATLAB.
  - I can solve a linear system <img src="https://latex.codecogs.com/gif.latex?A\boldsymbol{x}&space;=&space;\boldsymbol{b}" title="A\boldsymbol{x} = \boldsymbol{b}" />, or find a least-square solution of a underdetermined system, by calling `A\b`.
  - To find <img src="https://latex.codecogs.com/gif.latex?X" title="X" /> that minimizes <img src="https://latex.codecogs.com/gif.latex?||X\boldsymbol{w}&space;-&space;\boldsymbol{y}||^2" title="||X\boldsymbol{w} - \boldsymbol{y}||^2" />, one computes <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{w}&space;=&space;(X^TX)^{-1}X^T\boldsymbol{y}" title="\boldsymbol{w} = (X^TX)^{-1}X^T\boldsymbol{y}" />. In Julia, `X\y`.
    - http://web.stanford.edu/class/ee103/julia_slides/julia_least_squares_slides.pdf
- My code is comment, my code is concise, and Unicode is easy to code.
  - It feels great when it let me comment and name my variables in Unicode characters, especially with LaTeX-like abbreviations!
  - I can name my array of prediction <img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y}" title="\hat{y}" /> and each element <img src="https://latex.codecogs.com/gif.latex?\inline&space;y_i" title="y_i" />. No more `alpha_hat`: <img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{\alpha}" title="\hat{\alpha}" />.
- It is interactive and it can have a workspace, just like Python and R, but it races with C!
- Officially, it is said to be **"Fast, General, Dynamic, Technical, Optionally typed, Composable"**
