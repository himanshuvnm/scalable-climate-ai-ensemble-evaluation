# Method Note: Matrix-Free Covariance-Response Diagnostics

Data assimilation depends on how background-error covariance spreads information
from an observation through the atmospheric state. For high-dimensional climate
and weather states, explicitly forming the dense covariance matrix is usually
infeasible.

This workflow evaluates localized covariance responses through matrix-free
ensemble covariance actions. Given ensemble perturbations \(X\), the action of
the covariance matrix on a probe vector \(v\) is computed as

```math
Bv = X(X^\top v)/(N_e-1).
```

This allows one to visualize the covariance response around a localized
observation without forming \(B\).

The repository also includes query-conditioned weighting diagnostics. Instead of
using a static historical covariance baseline with equal weights over historical
cases, the conditioned response assigns larger weights to historical cases whose
descriptors are closer to the query case. This provides a scalable way to study
flow-dependent covariance-response structure.
