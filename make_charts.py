

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def try_np():
    try:
        import numpy as np
        return np
    except Exception:
        return None


def read_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def filter_rows_by_gen(rows, gen_min=None, gen_max=None):
    if gen_min is None and gen_max is None:
        return rows
    out = []
    for r in rows:
        try:
            g = int(float(r.get("gen", "0")))
        except Exception:
            continue
        if gen_min is not None and g < gen_min:
            continue
        if gen_max is not None and g > gen_max:
            continue
        out.append(r)
    return out


def col(rows, key, cast=float, default=None):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            out.append(default)
        else:
            out.append(cast(v))
    return out


def moving_avg(xs, w):
    if w <= 1:
        return xs[:]
    out, q, s = [], [], 0.0
    for x in xs:
        if x is None:
            out.append(None)
            continue
        q.append(float(x))
        s += float(x)
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def cummax(xs):
    out, m = [], None
    for x in xs:
        if x is None:
            out.append(m)
            continue
        m = x if m is None else max(m, x)
        out.append(m)
    return out


def gen_from_name(p: Path):
    m = re.search(r"gen(\d+)", p.name)
    return int(m.group(1)) if m else None


def latest_leq(run: Path, pattern: str, gen_max=None):
    items = sorted(run.glob(pattern))
    if gen_max is None:
        return items[-1] if items else None
    best = None
    for it in items:
        g = gen_from_name(it)
        if g is None:
            continue
        if g <= gen_max:
            best = it
    return best


def load_pop_snapshots(run: Path, gen_min=None, gen_max=None):
    np = try_np()
    if np is None:
        return []
    snaps = sorted(run.glob("pop_snapshot_gen*.npz"))
    out = []
    for s in snaps:
        g = gen_from_name(s)
        if g is None:
            continue
        if gen_min is not None and g < gen_min:
            continue
        if gen_max is not None and g > gen_max:
            continue
        data = np.load(s)
        out.append((s, data["X"], data["fitness"]))
    return out


def pca_2d(X):
    np = try_np()
    if np is None:
        return None
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T
    Z = Xc @ W
    return Z


def plot_gaussian_ellipse(Z, out_path: Path, title: str):
    np = try_np()
    if np is None:
        return
    mu = Z.mean(axis=0)
    C = np.cov(Z.T)
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linspace(0, 2*np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    A = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12)))
    ell = circle @ A.T + mu

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
    plt.plot(ell[:, 0], ell[:, 1], linewidth=2)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def simple_kmeans(Z, k=3, iters=30, seed=0):
    np = try_np()
    if np is None:
        return None
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    centers = Z[idx].copy()

    for _ in range(iters):
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            pts = Z[lab == j]
            if len(pts) > 0:
                new_centers[j] = pts.mean(axis=0)
        if (new_centers == centers).all():
            break
        centers = new_centers
    return lab


def make_single_run_figures(run: Path, gen_min=None, gen_max=None, tag=""):
    csv_path = run / "gen_stats.csv"
    if not csv_path.exists():
        return

    rows_all = read_csv(csv_path)
    rows = filter_rows_by_gen(rows_all, gen_min=gen_min, gen_max=gen_max)
    if not rows:
        print("No rows in requested gen range for", run)
        return

    def outname(base):
        return f"{base}__{tag}.png" if tag else f"{base}.png"

    g = col(rows, "gen", int)
    avg = col(rows, "avg_fitness", float)
    std = col(rows, "std_fitness", float)
    best = col(rows, "best_fitness", float)

    
    minus = [max(0.0, a - s) for a, s in zip(avg, std)]
    plus  = [a + s for a, s in zip(avg, std)]

    plt.figure()
    plt.plot(g, avg, label="average")
    plt.plot(g, minus, linestyle="--", label="-1 sd")
    plt.plot(g, plus, linestyle="--", label="+1 sd")
    plt.plot(g, best, label="best (per gen)")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("Population average and best fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / outname("fig_avg_best_std"), dpi=180)

    
    med = col(rows, "median_fitness", float)
    w = max(3, len(g) // 10)
    plt.figure()
    plt.plot(g, cummax(best), label="best-so-far (within zoom)")
    plt.plot(g, moving_avg(avg, w), label=f"avg (MA{w})")
    plt.plot(g, moving_avg(med, w), label=f"median (MA{w})")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("Learning curve (zoomed window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / outname("fig_learning_curve_pretty"), dpi=180)

    
    test_mean = col(rows, "test_mean_fitness", float, default=None)
    test_std = col(rows, "test_std_fitness", float, default=None)
    if any(v is not None for v in test_mean):
        xs, tm, lo, hi = [], [], [], []
        for gi, m, s in zip(g, test_mean, test_std):
            if m is None or s is None:
                continue
            xs.append(gi)
            tm.append(m)
            lo.append(m - s)
            hi.append(m + s)

        plt.figure()
        plt.plot(g, best, label="train best (per gen)")
        plt.plot(xs, tm, label="test mean (fixed seeds)")
        plt.fill_between(xs, lo, hi, alpha=0.2, label="test ±1 sd")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Train vs Test (zoomed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run / outname("fig_train_vs_test"), dpi=180)

    
    div = col(rows, "w_std_mean", float)
    plt.figure()
    plt.plot(g, div)
    plt.xlabel("generation")
    plt.ylabel("mean std(weights)")
    plt.title("Genome diversity over time (zoomed)")
    plt.tight_layout()
    plt.savefig(run / outname("fig_diversity_curve"), dpi=180)

    
    np = try_np()
    snap = latest_leq(run, "policy_snapshot_gen*.npz", gen_max=gen_max)
    if np is not None and snap is not None:
        data = np.load(snap)
        xs = data["xs"]; ys = data["ys"]; Z = data["Z"]
        plt.figure()
        plt.imshow(Z, origin="lower", aspect="auto",
                   extent=[ys.min(), ys.max(), xs.min(), xs.max()])
        plt.xlabel("vision1 (distance to pipe)")
        plt.ylabel("vision0 (above top pipe)")
        plt.title(f"Policy surface (latest in window) [{snap.name}]")
        plt.colorbar(label="network output")
        plt.tight_layout()
        plt.savefig(run / outname("fig_policy_surface_latest"), dpi=180)

    
    snaps = load_pop_snapshots(run, gen_min=gen_min, gen_max=gen_max)
    if snaps:
        plt.figure()
        stride = max(1, len(snaps)//6)
        for (p, X, y) in snaps[::stride]:
            plt.hist(y, bins=25, alpha=0.35, label=p.name.replace(".npz", ""))
        plt.xlabel("fitness")
        plt.ylabel("count")
        plt.title("Fitness distribution snapshots (zoomed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run / outname("fig_fitness_hist_snapshots"), dpi=180)

        p, X, y = snaps[-1]
        if np is not None and X.shape[0] >= 5 and X.shape[1] >= 2:
            Z = pca_2d(X)
            if Z is not None:
                plt.figure()
                sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=12, alpha=0.8)
                plt.xlabel("PC1"); plt.ylabel("PC2")
                plt.title(f"PCA of genomes (latest in window) [{p.name}]")
                plt.colorbar(sc, label="fitness")
                plt.tight_layout()
                plt.savefig(run / outname("fig_pca_genomes_latest"), dpi=180)

                plot_gaussian_ellipse(Z, run / outname("fig_pca_gaussian_ellipse"),
                                      title=f"Gaussian view of population (PCA) [{p.name}]")

                
                labels = None
                try:
                    from sklearn.mixture import GaussianMixture
                    K = 3 if Z.shape[0] >= 60 else 2
                    labels = GaussianMixture(n_components=K, random_state=0).fit_predict(Z)
                except Exception:
                    labels = simple_kmeans(Z, k=3, iters=40, seed=0)

                if labels is not None:
                    plt.figure()
                    for k in sorted(set(labels.tolist())):
                        pts = Z[labels == k]
                        plt.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, label=f"cluster {k}")
                    plt.title(f"Clusters in PCA space [{p.name}]")
                    plt.xlabel("PC1"); plt.ylabel("PC2")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(run / outname("fig_pca_clusters"), dpi=180)

            norms = (X**2).sum(axis=1) ** 0.5
            plt.figure()
            plt.scatter(norms, y, s=12, alpha=0.8)
            plt.xlabel("||weights||2")
            plt.ylabel("fitness")
            plt.title(f"Weight norm vs fitness [{p.name}]")
            plt.tight_layout()
            plt.savefig(run / outname("fig_weightnorm_vs_fitness"), dpi=180)

    
    np = try_np()
    ev = latest_leq(run, "eval_outputs_gen*.npz", gen_max=gen_max)
    if np is not None and ev is not None:
        data = np.load(ev)
        outs = data["outputs"]
        plt.figure()
        plt.hist(outs, bins=40)
        plt.xlabel("network output")
        plt.ylabel("count")
        plt.title(f"Network output histogram (latest eval in window) [{ev.name}]")
        plt.tight_layout()
        plt.savefig(run / outname("fig_output_hist_latest_eval"), dpi=180)

    
    sweep = run / "threshold_sweep.csv"
    if sweep.exists():
        rows2 = read_csv(sweep)
        thr = [float(r["threshold"]) for r in rows2]
        m = [float(r["test_mean_fitness"]) for r in rows2]
        s = [float(r["test_std_fitness"]) for r in rows2]
        lo = [mm - ss for mm, ss in zip(m, s)]
        hi = [mm + ss for mm, ss in zip(m, s)]
        plt.figure()
        plt.plot(thr, m)
        plt.fill_between(thr, lo, hi, alpha=0.2)
        plt.xlabel("threshold")
        plt.ylabel("test mean fitness")
        plt.title("Threshold sweep (hyperparameter curve)")
        plt.tight_layout()
        plt.savefig(run / outname("fig_threshold_sweep"), dpi=180)


def aggregate_runs(parent: Path, gen_min=None, gen_max=None, tag=""):
    runs = [p for p in parent.iterdir() if p.is_dir() and (p / "gen_stats.csv").exists()]
    if not runs:
        return

    curves_best = []
    curves_test = []
    max_len = 0

    for r in runs:
        rows = filter_rows_by_gen(read_csv(r / "gen_stats.csv"), gen_min=gen_min, gen_max=gen_max)
        if not rows:
            continue
        best = col(rows, "best_fitness", float)
        test = col(rows, "test_mean_fitness", float, default=None)
        curves_best.append(best)
        curves_test.append(test)
        max_len = max(max_len, len(best))

    if not curves_best:
        return

    def pad(xs, n): return xs + [None] * (n - len(xs))
    curves_best = [pad(c, max_len) for c in curves_best]
    curves_test = [pad(c, max_len) for c in curves_test]

    def mean_std_at(i, curves):
        vals = [c[i] for c in curves if c[i] is not None]
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        if len(vals) <= 1:
            return m, 0.0
        s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return m, s

    gens = list(range(1, max_len + 1))

    def outname(base):
        return f"{base}__{tag}.png" if tag else f"{base}.png"

    
    m_best, s_best = [], []
    for i in range(max_len):
        m, s = mean_std_at(i, curves_best)
        m_best.append(m); s_best.append(s)

    plt.figure()
    xs = [g for g, m in zip(gens, m_best) if m is not None]
    ys = [m for m in m_best if m is not None]
    lo = [m - s for m, s in zip(m_best, s_best) if m is not None]
    hi = [m + s for m, s in zip(m_best, s_best) if m is not None]
    plt.plot(xs, ys, label="mean train best")
    plt.fill_between(xs, lo, hi, alpha=0.2, label="±1 sd")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title("Aggregate learning curve across runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(parent / outname("fig_aggregate_learning_curve"), dpi=180)

    
    m_test, s_test = [], []
    for i in range(max_len):
        m, s = mean_std_at(i, curves_test)
        m_test.append(m); s_test.append(s)

    if any(m is not None for m in m_test):
        plt.figure()
        xs = [g for g, m in zip(gens, m_test) if m is not None]
        ys = [m for m in m_test if m is not None]
        lo = [m - s for m, s in zip(m_test, s_test) if m is not None]
        hi = [m + s for m, s in zip(m_test, s_test) if m is not None]
        plt.plot(xs, ys, label="mean test fitness (fixed seeds)")
        plt.fill_between(xs, lo, hi, alpha=0.2, label="±1 sd")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Aggregate test curve across runs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(parent / outname("fig_aggregate_test_curve"), dpi=180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs/exp1")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--gen_min", type=int, default=None)
    ap.add_argument("--gen_max", type=int, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    p = Path(args.path)

    if args.aggregate:
        aggregate_runs(p, gen_min=args.gen_min, gen_max=args.gen_max, tag=args.tag)
        for r in sorted([x for x in p.iterdir() if x.is_dir()]):
            make_single_run_figures(r, gen_min=args.gen_min, gen_max=args.gen_max, tag=args.tag)
        print("Wrote aggregate figures into:", p)
        return

    make_single_run_figures(p, gen_min=args.gen_min, gen_max=args.gen_max, tag=args.tag)
    print("Wrote figures into:", p)


if __name__ == "__main__":
    main()
