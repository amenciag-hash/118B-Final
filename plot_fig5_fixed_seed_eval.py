import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to gen_stats.csv")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: alongside csv)")
    ap.add_argument("--max_gen", type=int, default=0, help="If >0, only plot gens <= max_gen")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.out) if args.out else (csv_path.parent / "fig5_fixed_seed_eval.png")

    gens = []
    mean_fit = []
    std_fit = []
    mean_pipes = []
    std_pipes = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("CSV has no header row (field names).")

        
        g_col = "gen" if "gen" in r.fieldnames else r.fieldnames[0]
        mf_col = "test_mean_fitness" if "test_mean_fitness" in r.fieldnames else None
        sf_col = "test_std_fitness" if "test_std_fitness" in r.fieldnames else None
        mp_col = "test_mean_pipes" if "test_mean_pipes" in r.fieldnames else None
        sp_col = "test_std_pipes" if "test_std_pipes" in r.fieldnames else None

        if mf_col is None and mp_col is None:
            raise ValueError(
                "Could not find fixed-seed eval columns. "
                "Expected test_mean_fitness/test_std_fitness (or test_mean_pipes/test_std_pipes)."
            )

        for row in r:
            g = to_float(row.get(g_col))
            if g is None:
                continue
            g = int(g)
            if args.max_gen > 0 and g > args.max_gen:
                continue

            
            mf = to_float(row.get(mf_col)) if mf_col else None
            sf = to_float(row.get(sf_col)) if sf_col else None

            
            mp = to_float(row.get(mp_col)) if mp_col else None
            sp = to_float(row.get(sp_col)) if sp_col else None

            
            if mf is not None:
                gens.append(g)
                mean_fit.append(mf)
                std_fit.append(sf if sf is not None else 0.0)
            elif mp is not None:
                gens.append(g)
                mean_pipes.append(mp)
                std_pipes.append(sp if sp is not None else 0.0)

    if len(gens) == 0:
        raise ValueError("No fixed-seed eval rows found. Did you run with --eval_every 1 and --eval_seeds ... ?")

    plt.figure()
    if len(mean_fit) == len(gens) and len(mean_fit) > 0:
        lo = [m - s for m, s in zip(mean_fit, std_fit)]
        hi = [m + s for m, s in zip(mean_fit, std_fit)]
        plt.plot(gens, mean_fit, label="test mean fitness (fixed seeds)")
        plt.fill_between(gens, lo, hi, alpha=0.2, label="± 1 std")
        plt.ylabel("Fitness (lifespan)")
        plt.title("Figure 5: Fixed-seed evaluation over generations")
    else:
        lo = [m - s for m, s in zip(mean_pipes, std_pipes)]
        hi = [m + s for m, s in zip(mean_pipes, std_pipes)]
        plt.plot(gens, mean_pipes, label="test mean pipes (fixed seeds)")
        plt.fill_between(gens, lo, hi, alpha=0.2, label="± 1 std")
        plt.ylabel("Pipes passed")
        plt.title("Figure 5: Fixed-seed evaluation (pipes) over generations")

    plt.xlabel("Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
