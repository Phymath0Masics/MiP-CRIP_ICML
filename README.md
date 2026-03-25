# IAMP vs MiP-CRIP (SK Benchmark)

Small benchmark project comparing two solvers on the Sherrington-Kirkpatrick (SK) Ising model:

- `IAMP` (Incremental Approximate Message Passing) [[1]](#references)
- `MiP-CRIP` (Minima Preserving Continuous Relaxation)

## Files

- `benchmark_SK.py` : benchmark runner and summary table output
- `iamp_sk_solver.py` : IAMP implementation
- `mip_crip.py` : MiP-CRIP implementation
- `environment.yml` : minimal Conda environment

## Quick Start

```bash
conda env create -f environment.yml
conda activate iamp-vs-mip-crip
python benchmark_SK.py
```

## Fixed Parameters

The benchmark uses the following fine-tued hyperparameters, which are fixed throughout every test case so both methods are compared under a consistent setup.

- IAMP: `beta=6.0`, `delta=0.02`, `n_restarts=3`.
- MiP-CRIP: `T=10`, `K=200`, `alpha=0.000014996`($\alpha$), `beta=0.001`($\beta$), `lambda_=0.0707`($\lambda$), `step=1.0`($\tau$), `beta1=0.09` (1st moment for ADAM), `beta2=0.999`(2nd moment for ADAM), `eps=1e-8`, `sigma_noise=1e-3`.

- The elements $J_{ij}$ is the SK model are rounded to 5th decimal places so that $lsb = 10^{-5}$ and we get thebound $\gamma_0 = 10^{-5}$, satisfying $3\beta\lambda^2 < \alpha < \beta\lambda^2 + \gamma_0$ for the MiP-CRIP parameters.

## Notes

The benchmark uses both type of SK models:

- **Gaussian Orthogonal Ensemble (GOE)** [[1]](#references): $J \in \mathbb{R}^{n\times n}$ where $J_{ij} = J_{ji} \sim \mathcal{N}[0, 1/n]$ for $i\neq j$ and $J_{ii} \sim \mathcal{N}[0, 2/n]$.
- **Standard**: $J_{ij} \sim \mathcal{N}[0, 1]$ with $J_{ij} = J_{ji}$ and zero diagonal.

## References

- "Optimization of the Sherrington-Kirkpatrick Hamiltonian." *arXiv:1812.10897v2*.
