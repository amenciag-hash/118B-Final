"""
analyze_run_pretty.py
---------------------
Makes cleaner PRML-style plots from runs/<run>/gen_stats.csv

Why your plot looked messy:
- "best per generation" is super spiky (one lucky agent hits max_steps).
- avg/median jump because environment changes every generation (new random pipes).
Fixes:
- plot BEST-SO-FAR (cumulative max) instead of best-per-gen
- smooth avg/median with a moving average
- add an IQR band (25%..75%) if you also saved pop snapshots (optional)

Run:
  python analyze_run_pretty.py runs/run1

Outputs:
  fig_learning_curve_pretty.png
  fig_learning_curve_best_so_far.png
  fig_fraction_capped.png  (how often fitness hits the cap ~ max_steps)
"""

import sys
from pathlib import Path
import csv

import matplotlib.pyplot as plt


def read_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(rows, key):
    return [float(r[key]) for r in rows]


def to_int(rows, key):
    return [int(float(r[key])) for r in rows]


def moving_avg(xs, w):
    if w <= 1:
        return xs[:]
    out = []
    s = 0.0
    q = []
    for x in xs:
        q.append(float(x))
        s += float(x)
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def cummax(xs):
    out = []
    m = None
    for x in xs:
        if m is None or x > m:
            m = x
        out.append(m)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_run_pretty.py <run_folder>")
        sys.exit(1)

    run = Path(sys.argv[1])
    csv_path = run / "gen_stats.csv"
    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        sys.exit(1)

    rows = read_csv(csv_path)
    gen = to_int(rows, "gen")

    best = to_float(rows, "best_fitness")
    avg = to_float(rows, "avg_fitness")
    med = to_float(rows, "median_fitness")

    
    cap = max(best) if best else 0.0

    w = max(5, len(gen) // 30)  
    best_so_far = cummax(best)
    avg_s = moving_avg(avg, w)
    med_s = moving_avg(med, w)

    
    plt.figure()
    plt.plot(gen, best_so_far, label="best-so-far")
    plt.plot(gen, avg_s, label=f"avg (MA{w})")
    plt.plot(gen, med_s, label=f"median (MA{w})")
    plt.xlabel("generation")
    plt.ylabel("fitness (lifespan)")
    plt.title("Learning curve (smoothed + best-so-far)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / "fig_learning_curve_pretty.png", dpi=180)

    
    plt.figure()
    plt.plot(gen, best, label="best (per gen)")
    plt.plot(gen, best_so_far, label="best-so-far")
    plt.xlabel("generation")
    plt.ylabel("fitness (lifespan)")
    plt.title("Best-per-generation is spiky; best-so-far is stable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / "fig_learning_curve_best_so_far.png", dpi=180)

    
    
    if cap > 0:
        frac_capped = [1.0 if (b >= cap) else 0.0 for b in best]
        frac_capped_s = moving_avg(frac_capped, w)
        plt.figure()
        plt.plot(gen, frac_capped_s)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("generation")
        plt.ylabel("fraction of generations where best hit cap")
        plt.title("How often best fitness hits max_steps (cap)")
        plt.tight_layout()
        plt.savefig(run / "fig_fraction_capped.png", dpi=180)

    print("Wrote figures into:", run)


if __name__ == "__main__":
    main()
