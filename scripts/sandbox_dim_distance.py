"""Sandbox: how pairwise distances grow with dimension in the unit cube [0,1]^d.

Run:  python sandbox_dim_distance.py

Intuition this builds
---------------------
For two points u, v drawn uniformly in [0,1]^d, each coordinate contributes
    E[(u_i - v_i)^2] = 1/6
independently, so the *squared* distance is a sum of d iid terms:
    E[||u - v||^2] = d/6      ->   typical distance ~ sqrt(d/6)  (grows with sqrt(d))
And because it's a sum of d iid terms, the distribution CONCENTRATES:
    std(distance) / mean(distance)  ~  O(1/sqrt(d))  ->  0
i.e. in high d, (a) every pair is far apart, and (b) all pairs are at *almost the
same* distance. That second part is the "curse" — points become equidistant, so a
stationary kernel can't tell near from far unless the lengthscale grows ~sqrt(d).
That sqrt(d) is exactly what the Hvarfner / Stuyver lengthscale priors bake in.
"""
import numpy as np

THEO_MEAN_PER_DIM_SQ = 1.0 / 6.0  # E[(u_i - v_i)^2] for u_i,v_i ~ U[0,1]


def distance_stats(d, n_points=2000, seed=0):
    """Sample n_points uniformly in [0,1]^d and summarise pairwise distances."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, d))

    # Squared pairwise distances via ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v.
    # This is (n x n) memory regardless of d (no (n, n, d) broadcast blowup).
    sq = (X ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)                          # clamp tiny negatives
    iu = np.triu_indices(n_points, k=1)
    dist = np.sqrt(D2[iu])

    # Nearest-neighbour distance per point (min over off-diagonal).
    np.fill_diagonal(D2, np.inf)
    nn = np.sqrt(D2.min(axis=1))

    mean, std = dist.mean(), dist.std()
    return {
        "d": d,
        "mean_dist": mean,
        "std_dist": std,
        "min_dist": dist.min(),
        "max_dist": dist.max(),
        "nn_dist": nn.mean(),                       # mean nearest-neighbour distance
        "theory_sqrt(d/6)": np.sqrt(d * THEO_MEAN_PER_DIM_SQ),
        "mean/sqrt(d)": mean / np.sqrt(d),          # should approach sqrt(1/6)=0.408
        "concentration std/mean": std / mean,       # should shrink ~ 1/sqrt(d)
    }


def main():
    dims = [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 1280]  # 1280 = ESM2 dim
    rows = [distance_stats(d) for d in dims]

    cols = ["d", "mean_dist", "theory_sqrt(d/6)", "nn_dist", "min_dist",
            "max_dist", "mean/sqrt(d)", "concentration std/mean"]
    hdr = "".join(f"{c:>22}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("".join(f"{r[c]:>22.4f}" if c != "d" else f"{r[c]:>22d}" for c in cols))

    print(
        "\nRead-off:\n"
        "  * mean_dist tracks sqrt(d/6) almost exactly -> distance GROWS ~ sqrt(d).\n"
        "  * mean/sqrt(d) flattens to ~0.408 (=sqrt(1/6)) -> the sqrt(d) law.\n"
        "  * std/mean shrinks toward 0 -> distances CONCENTRATE (near==far).\n"
        "  * even nearest-neighbour distance grows -> the cube empties out.\n"
        "\nWhy the lengthscale prior scales with sqrt(d): to keep r/lengthscale in a\n"
        "useful range as r ~ sqrt(d) grows, lengthscale must grow ~ sqrt(d) too.\n"
        "Hvarfner median lengthscale ~ 4.1*sqrt(d); on [0,1]^d that gives\n"
        "r/lengthscale = sqrt(d/6)/(4.1*sqrt(d)) = 0.10  -- constant in d (the point!)."
    )

    # Optional plot (saved to file; skipped if matplotlib is unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds = np.array([r["d"] for r in rows])
        means = np.array([r["mean_dist"] for r in rows])
        conc = np.array([r["concentration std/mean"] for r in rows])

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ds, means, "o-", label="empirical mean dist")
        ax[0].plot(ds, np.sqrt(ds / 6), "--", label=r"$\sqrt{d/6}$")
        ax[0].set_xlabel("dimension d"); ax[0].set_ylabel("mean pairwise distance")
        ax[0].set_title("distance grows ~ sqrt(d)"); ax[0].legend()

        ax[1].plot(ds, conc, "o-")
        ax[1].plot(ds, conc[0] / np.sqrt(ds / ds[0]), "--", label=r"$\propto 1/\sqrt{d}$")
        ax[1].set_xlabel("dimension d"); ax[1].set_ylabel("std/mean of distances")
        ax[1].set_title("distances concentrate (near == far)"); ax[1].legend()

        fig.tight_layout()
        out = "sandbox_dim_distance.png"
        fig.savefig(out, dpi=120)
        print(f"\nSaved plot to {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
