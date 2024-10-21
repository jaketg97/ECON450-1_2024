## GMM Definition

- Touch on this later, but this relaxed requirement is key. Just need moments; much easier than fully defining the DGP.
- "Moments" are very ethereal. Easier to map to reduced form(s) of structural models. That is, imagine you have a structural model of how your data is generated that depends on a set of exogenous parameters $\theta$ that you're interested in estimating. You can manipulate that structural model to derive reduced form equations that map to moments (this is what you're doing in your PSET). 
- Point is you end up with model implied moments $m(x|\theta)$ you can compare to data moments $m(x)$. Choose $\theta$ to minimize the distance between them.
- Theoretical mapping...
$$
g(X, \theta) = m(X|\theta) - m(X) \rightarrow E[g(X,\theta)] = 0
$$

## Weighting matrix

- Optimal weighting matrix is the variance covariance matrix of the moment condition errors at the optimal parameter values (I'll formally define that in a little bit, but hopefully that's already clear). Main thing is this definition is circular! So it's clear that true identification of it would require a fixed point theorem. 
- In practice use Identity Matrix...
- Or two step (just one iteration of a fixed point algorithm).

## Strengths

- ML is asymptotically efficient; going to produce smallest SEs.
- But that's because of assumptions! To maximize likelihood you need to have likelihood; that means you're fully specifying the DGP. 
- GMM doesn't need this! All you need is moments $\geq$ parameters. Way more flexible! Incredibly well suited to structural estimation.

## Code

- First write fnct to generate truncated normal pdf values for xvals. Truncating N(mu, sigma) at lower bound and upper bound; so inflate PDF of N(mu, sigma) based on what's being truncated. 
- data_moments are obviously just the mean and variance of the data.
- model_moments are the mean and variance implied by truncating N(mu, sigma) at 0, 450. 
- Calculate objective function finally, and then optimize. 