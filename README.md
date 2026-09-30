# phi-diffusion-hg

Reference implementation of Φ-SIS, the SIS instantiation of the exposure-based
within-hyperedge dynamics framework described in the paper "Exposure-Based Within-Hyperedge Dynamics for Diffusion on Hypergraphs", currently under review.

## Stucture of the dataset

    data/   hypergraph datasets (one hyperedge per line, space-separated integer node ids)
    src/    phisis.py, the simulator

## Requirements

Python 3.9+, numpy, scipy (`pip install -r requirements.txt`).

## Usage

    python src/phisis.py data/Algebra.txt --beta 0.01 --gamma 0.1 --delta 0.25 --T 30 --K 2 --seeds random --f lin --runs 300

Options

| flag | meaning | default |
|---|---|---|
| `--beta` | transmissibility β | 0.01 |
| `--gamma` | recovery probability γ | 0 |
| `--delta` | memory decay δ ∈ [0,1] (δ=1 gives the memoryless RP) | 0 |
| `--T` | time horizon | 30 |
| `--K` | number of seeds | 2 |
| `--seeds` | `random`, `degree`, `hyperdegree`, or explicit ids, e.g. `--seeds 12,45` | random |
| `--f` | reinforcement function: `lin` (x), `sub` (log(1+x)), `sat` (x/(1+x)), `refr` (x e^{-αx}) | lin |
| `--alpha` | α of the refractory function | 0.5 |
| `--immunize` | node ids removed from the process (they neither get infected nor deposit exposure) | – |
| `--runs` | Monte Carlo runs | 300 |
| `--rng` | random seed | 0 |
| `--out` | write mean/std curve, per-run final fractions and seeds to a json file | – |

The program prints the mean and standard deviation of the infected fraction at
each step t = 0, …, T over the runs.