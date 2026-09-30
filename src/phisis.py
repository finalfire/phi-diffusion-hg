import argparse
import json
import sys

import numpy as np
from scipy import sparse


def read_hypergraph(path):
    edges = []
    with open(path) as fh:
        for line in fh:
            toks = line.split()
            if toks:
                edges.append([int(t) for t in toks])
    ids = sorted({v for e in edges for v in e})
    idx = {v: i for i, v in enumerate(ids)}
    rows, cols = [], []
    for j, e in enumerate(edges):
        for v in set(e):
            rows.append(idx[v])
            cols.append(j)
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(edges)))
    return B, np.array(ids)


REINFORCEMENT = {
    "lin": lambda x, a: x,
    "sub": lambda x, a: np.log1p(x),
    "sat": lambda x, a: x / (1.0 + x),
    "refr": lambda x, a: x * np.exp(-a * x),
}


def pick_seeds(B, K, mode, rng):
    n = B.shape[0]
    if mode == "random":
        return rng.choice(n, K, replace=False)
    if mode == "hyperdegree":
        score = np.asarray(B.sum(axis=1)).ravel()
    elif mode == "degree":
        A = (B @ B.T).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        score = np.diff(A.indptr)
    else:
        raise ValueError(mode)
    return np.argsort(-score, kind="stable")[:K]


def run(B, seeds, beta, gamma, delta, T, f, alpha, rng, immunized=None):
    n, m = B.shape
    Bt = B.T.tocsr()
    indptr, col = B.indptr, B.indices
    row = np.repeat(np.arange(n), np.diff(indptr))
    eta = np.zeros(len(col))

    infected = np.zeros(n, dtype=bool)
    infected[seeds] = True
    blocked = np.zeros(n, dtype=bool)
    if immunized is not None:
        blocked[immunized] = True
        infected &= ~blocked

    curve = np.empty(T + 1)
    curve[0] = infected.mean()
    for t in range(T):
        active = (Bt @ infected.astype(np.int8)) > 0
        eta *= 1.0 - delta
        eta += (~infected[row] & ~blocked[row] & active[col]).astype(float)

        lam = beta * np.add.reduceat(f(eta, alpha), indptr[:-1])
        lam[np.diff(indptr) == 0] = 0.0
        p_inf = 1.0 - np.exp(-lam)

        u = rng.random(n)
        new = np.where(infected, u >= gamma, u < p_inf)
        new &= ~blocked
        infected = new
        curve[t + 1] = infected.mean()
    return curve, infected


def main():
    ap = argparse.ArgumentParser(description="Phi-SIS: SIS diffusion with exposure-based within-hyperedge dynamics")
    ap.add_argument("dataset", help="text file, one hyperedge per line, space-separated integer node ids")
    ap.add_argument("--beta", type=float, default=0.01, help="transmissibility")
    ap.add_argument("--gamma", type=float, default=0.0, help="recovery probability")
    ap.add_argument("--delta", type=float, default=0.0, help="memory decay in [0,1]")
    ap.add_argument("--T", type=int, default=30, help="time horizon")
    ap.add_argument("--K", type=int, default=2, help="number of seeds")
    ap.add_argument("--seeds", default="random",
                    help="random | degree | hyperdegree | comma-separated node ids")
    ap.add_argument("--f", choices=list(REINFORCEMENT), default="lin", help="reinforcement function")
    ap.add_argument("--alpha", type=float, default=0.5, help="rate of the refractory function")
    ap.add_argument("--immunize", default="", help="comma-separated node ids removed from the process")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--rng", type=int, default=0, help="random seed")
    ap.add_argument("--out", help="write results as json")
    args = ap.parse_args()

    B, ids = read_hypergraph(args.dataset)
    lookup = {v: i for i, v in enumerate(ids)}
    f = REINFORCEMENT[args.f]
    rng = np.random.default_rng(args.rng)

    fixed = None
    if args.seeds not in ("random", "degree", "hyperdegree"):
        fixed = np.array([lookup[int(s)] for s in args.seeds.split(",")])
    immunized = None
    if args.immunize:
        immunized = np.array([lookup[int(s)] for s in args.immunize.split(",")])

    curves = []
    seed_log = []
    for _ in range(args.runs):
        s = fixed if fixed is not None else pick_seeds(B, args.K, args.seeds, rng)
        c, _ = run(B, s, args.beta, args.gamma, args.delta, args.T, f, args.alpha, rng, immunized)
        curves.append(c)
        seed_log.append(ids[s].tolist())
    curves = np.array(curves)
    mean, std = curves.mean(0), curves.std(0)

    print("t\tmean\tstd")
    for t in range(args.T + 1):
        print(f"{t}\t{mean[t]:.4f}\t{std[t]:.4f}")
    print(f"# n={B.shape[0]} m={B.shape[1]} final={mean[-1]:.4f}+-{std[-1]:.4f}", file=sys.stderr)

    if args.out:
        res = {
            "dataset": args.dataset,
            "params": {k: v for k, v in vars(args).items() if k not in ("dataset", "out")},
            "n": int(B.shape[0]), "m": int(B.shape[1]),
            "mean": mean.tolist(), "std": std.tolist(),
            "final": curves[:, -1].tolist(),
            "seeds": seed_log if fixed is None else seed_log[0],
        }
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
