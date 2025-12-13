"""
fast_train_plus.py
------------------
Fast headless evolution simulator (uses your repo's Player + Brain) AND logs PRML-style data.

Drop this file into your repo folder next to:
  player.py, brain.py, node.py, connection.py, config.py

Run example:
  python fast_train_plus.py --gens 100 --pop 50 --max_steps 4000 --logdir runs/exp1 --save_pop_every 5 --save_policy_every 10 --eval_every 5 --eval_seeds 0,1,2,3,4

Outputs in --logdir:
  - run_config.json
  - gen_stats.csv                     (now includes train + test metrics)
  - best_genome_weights.csv
  - pop_snapshot_genXXXX.npz          (optional; needs numpy)
  - policy_snapshot_genXXXX.npz       (optional; needs numpy)

Notes:
- No window, no drawing, no FPS cap.
- Physics/collisions come from Player.update(), Player.bird_flap(), Player.pipe_collision() etc.
- We avoid Player.look() because it draws lines; we compute vision numerically.
"""

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")

import argparse
import csv
import json
import random
import time
from pathlib import Path

import pygame

import config
from player import Player





SCREEN_W = 500
SCREEN_H = 500
GROUND_Y = 500          
BIRD_X = 50             
PIPE_W = 15             
PIPE_OPENING = 110      
PIPE_SPEED = 2.5        

SPAWN_X = SCREEN_W


ELITE_FRAC = 0.10
SURVIVOR_FRAC = 0.25


def try_np():
    try:
        import numpy as np
        return np
    except Exception:
        return None


class Pipe:
    """Keep exactly ONE pipe in config.pipes so Player.pipe_collision() works."""
    __slots__ = ("x", "bottom_h", "top_h", "top_rect", "bottom_rect", "passed", "_rng")

    def __init__(self, x: float, rng: random.Random):
        self._rng = rng
        self.x = x
        self.bottom_h = rng.randint(10, 300)
        self.top_h = GROUND_Y - self.bottom_h - PIPE_OPENING

        self.top_rect = pygame.Rect(int(self.x), 0, PIPE_W, int(self.top_h))
        self.bottom_rect = pygame.Rect(int(self.x), int(GROUND_Y - self.bottom_h), PIPE_W, int(self.bottom_h))
        self.passed = False

    def step(self):
        self.x -= PIPE_SPEED
        xi = int(self.x)
        self.top_rect.x = xi
        self.bottom_rect.x = xi
        if (not self.passed) and ((self.x + PIPE_W) < BIRD_X):
            self.passed = True

    def respawn(self):
        
        self.x = SPAWN_X
        self.bottom_h = self._rng.randint(10, 300)
        self.top_h = GROUND_Y - self.bottom_h - PIPE_OPENING
        self.top_rect = pygame.Rect(int(self.x), 0, PIPE_W, int(self.top_h))
        self.bottom_rect = pygame.Rect(int(self.x), int(GROUND_Y - self.bottom_h), PIPE_W, int(self.bottom_h))
        self.passed = False


def closest_pipe():
    for p in config.pipes:
        if not getattr(p, "passed", False):
            return p
    return config.pipes[0]


def compute_vision(player: Player, pipe: Pipe):
    
    player.vision[0] = max(0, player.rect.center[1] - pipe.top_rect.bottom) / SCREEN_H
    player.vision[1] = max(0, pipe.x - player.rect.center[0]) / SCREEN_W
    player.vision[2] = max(0, pipe.bottom_rect.top - player.rect.center[1]) / SCREEN_H


def ensure_tracking_fields(pl: Player):
    
    if not hasattr(pl, "pipes_passed"):
        pl.pipes_passed = 0
    if not hasattr(pl, "flap_count"):
        pl.flap_count = 0
    if not hasattr(pl, "abs_gap_err_sum"):
        pl.abs_gap_err_sum = 0.0
    if not hasattr(pl, "alive_steps"):
        pl.alive_steps = 0


def gap_center_y(pipe: Pipe) -> float:
    return pipe.top_rect.bottom + (PIPE_OPENING / 2.0)


def genome_vector(pl: Player):
    ws = []
    for c in getattr(pl.brain, "connections", []):
        w = getattr(c, "weight", None)
        if w is None:
            w = getattr(c, "w", None)
        if w is None:
            continue
        ws.append(float(w))
    return ws


def set_genome_vector(pl: Player, ws):
    
    i = 0
    for c in getattr(pl.brain, "connections", []):
        if hasattr(c, "weight"):
            c.weight = float(ws[i])
        elif hasattr(c, "w"):
            c.w = float(ws[i])
        i += 1
        if i >= len(ws):
            break


def diversity_stats(players):
    np = try_np()
    mats = [genome_vector(p) for p in players]
    if not mats or not mats[0]:
        return {"w_dim": 0, "w_var_mean": 0.0, "w_std_mean": 0.0}

    d = len(mats[0])
    for i in range(len(mats)):
        if len(mats[i]) != d:
            mats[i] = (mats[i] + [0.0] * d)[:d]

    if np is None:
        N = len(mats)
        means = [sum(m[j] for m in mats) / N for j in range(d)]
        vars_ = []
        for j in range(d):
            v = sum((m[j] - means[j]) ** 2 for m in mats) / max(1, (N - 1))
            vars_.append(v)
        return {
            "w_dim": d,
            "w_var_mean": sum(vars_) / d,
            "w_std_mean": (sum((v ** 0.5) for v in vars_) / d),
        }

    import numpy as np2
    X = np2.array(mats, dtype=float)
    return {
        "w_dim": int(X.shape[1]),
        "w_var_mean": float(np2.var(X, axis=0).mean()),
        "w_std_mean": float(np2.std(X, axis=0).mean()),
    }


def flap_rate(p):
    return (p.flap_count / p.alive_steps) if p.alive_steps else 0.0


def gap_err(p):
    return (p.abs_gap_err_sum / p.alive_steps) if p.alive_steps else 0.0


def simulate_generation(players, max_steps: int, decision_threshold: float, rng: random.Random, collect_outputs: bool=False):
    """
    Simulate one generation with shared pipes (same environment per generation).
    Returns:
        outputs (list[float]) if collect_outputs else None
    """
    config.pipes = [Pipe(SPAWN_X, rng)]
    ground_rect = pygame.Rect(0, GROUND_Y, SCREEN_W, 5)

    for pl in players:
        ensure_tracking_fields(pl)

    alive = len(players)
    steps = 0
    outputs = [] if collect_outputs else None

    while alive > 0 and steps < max_steps:
        steps += 1

        pipe = closest_pipe()
        prev_passed = pipe.passed
        pipe.step()

        
        if (not prev_passed) and pipe.passed:
            for pl in players:
                if pl.alive:
                    pl.pipes_passed += 1

        
        if pipe.x + PIPE_W < 0:
            config.pipes[0].respawn()
            pipe = config.pipes[0]

        gap_y = gap_center_y(pipe)

        for pl in players:
            if not pl.alive:
                continue

            compute_vision(pl, pipe)

            pl.abs_gap_err_sum += abs(pl.rect.center[1] - gap_y)
            pl.alive_steps += 1

            out = pl.brain.feed_forward(pl.vision)
            pl.decision = out
            if collect_outputs:
                outputs.append(float(out))

            if out > decision_threshold:
                before_vel = pl.vel
                before_flap = pl.flap
                pl.bird_flap()
                if (not before_flap) and (pl.vel < before_vel):
                    pl.flap_count += 1

            pl.update(ground_rect)

            if not pl.alive:
                alive -= 1

    for pl in players:
        pl.calculate_fitness()

    return outputs


def evolve(players, pop_size: int):
    players.sort(key=lambda p: p.fitness, reverse=True)

    elite_n = max(1, int(pop_size * ELITE_FRAC))
    parent_pool_n = max(elite_n, int(pop_size * SURVIVOR_FRAC))
    elites = players[:elite_n]
    parent_pool = players[:parent_pool_n]

    next_gen = [e.clone() for e in elites]

    while len(next_gen) < pop_size:
        parent = random.choice(parent_pool)
        child = parent.clone()
        child.brain.mutate()
        next_gen.append(child)

    return next_gen, players[0]


def write_best_weights_csv(path: Path, best: Player):
    ws = genome_vector(best)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "weight"])
        for i, val in enumerate(ws):
            w.writerow([i, val])


def maybe_save_population_snapshot(logdir: Path, gen: int, players):
    np = try_np()
    if np is None:
        return

    mats = [genome_vector(p) for p in players]
    if not mats or not mats[0]:
        return

    d = len(mats[0])
    for i in range(len(mats)):
        if len(mats[i]) != d:
            mats[i] = (mats[i] + [0.0] * d)[:d]

    X = np.array(mats, dtype=float)
    y = np.array([p.fitness for p in players], dtype=float)
    out = logdir / f"pop_snapshot_gen{gen:04d}.npz"
    np.savez_compressed(out, X=X, fitness=y)


def maybe_save_policy_snapshot(logdir: Path, gen: int, best: Player, grid_n: int):
    np = try_np()
    if np is None:
        return

    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    v2_fixed = 0.5
    Z = np.zeros((grid_n, grid_n), dtype=float)

    for i, v0 in enumerate(xs):
        for j, v1 in enumerate(ys):
            Z[i, j] = best.brain.feed_forward([float(v0), float(v1), float(v2_fixed)])

    out = logdir / f"policy_snapshot_gen{gen:04d}.npz"
    np.savez_compressed(out, xs=xs, ys=ys, Z=Z, v2_fixed=v2_fixed)


def evaluate_best_on_seeds(best_weights, seeds, max_steps, threshold):
    """
    Evaluate ONE genome on multiple fixed pipe seeds.
    Returns: dict with mean/std fitness and mean/std pipes, plus output histogram data.
    """
    fits = []
    pipes = []
    all_out = []

    for s in seeds:
        rng = random.Random(int(s))

        pl = Player()
        ensure_tracking_fields(pl)
        set_genome_vector(pl, best_weights)

        
        outs = simulate_generation([pl], max_steps=max_steps, decision_threshold=threshold, rng=rng, collect_outputs=True)
        fits.append(float(pl.fitness))
        pipes.append(float(getattr(pl, "pipes_passed", 0)))
        if outs:
            all_out.extend(outs)

    
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def std(xs):
        if len(xs) <= 1: return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    return {
        "test_mean_fitness": mean(fits),
        "test_std_fitness": std(fits),
        "test_mean_pipes": mean(pipes),
        "test_std_pipes": std(pipes),
        "outputs": all_out
    }


def parse_seeds(s: str):
    
    s = s.replace(",", " ").strip()
    if not s:
        return []
    return [int(x) for x in s.split() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--threshold", type=float, default=0.73)

    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--save_pop_every", type=int, default=0)
    parser.add_argument("--save_policy_every", type=int, default=0)
    parser.add_argument("--policy_grid", type=int, default=35)

    
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate best genome on fixed seeds every N generations (0 disables).")
    parser.add_argument("--eval_seeds", type=str, default="0,1,2,3,4", help="Comma/space-separated pipe seeds used for test evaluation.")
    parser.add_argument("--threshold_sweep", type=str, default="", help="Optional: evaluate best genome with thresholds e.g. '0.4,0.5,0.6,0.7,0.8,0.9' at end.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        pygame.init()
    except Exception:
        
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.init()

    config.window = None

    
    if args.logdir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.logdir = str(Path("runs") / ts)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    eval_seeds = parse_seeds(args.eval_seeds)

    
    cfg = {
        "pop": args.pop,
        "gens": args.gens,
        "max_steps": args.max_steps,
        "threshold": args.threshold,
        "PIPE_SPEED": PIPE_SPEED,
        "ELITE_FRAC": ELITE_FRAC,
        "SURVIVOR_FRAC": SURVIVOR_FRAC,
        "seed": args.seed,
        "policy_grid": args.policy_grid,
        "save_pop_every": args.save_pop_every,
        "save_policy_every": args.save_policy_every,
        "eval_every": args.eval_every,
        "eval_seeds": eval_seeds,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (logdir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    gen_stats_path = logdir / "gen_stats.csv"
    with gen_stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "gen",
            "best_fitness", "avg_fitness", "median_fitness", "std_fitness",
            "best_pipes_passed", "avg_pipes_passed",
            "best_flap_rate", "avg_flap_rate",
            "best_gap_err", "avg_gap_err",
            "w_dim", "w_var_mean", "w_std_mean",
            "frac_best_capped",
            "test_mean_fitness", "test_std_fitness",
            "test_mean_pipes", "test_std_pipes",
            "elapsed_sec"
        ])

    players = [Player() for _ in range(args.pop)]

    t0 = time.perf_counter()
    best_ever = None
    last_outputs_for_hist = []

    for g in range(1, args.gens + 1):
        
        train_rng = random.Random((args.seed or 0) * 1000003 + g * 9176 + 123)
        simulate_generation(players,
                             max_steps=args.max_steps, 
                             decision_threshold=args.threshold,
                               rng=train_rng)

        fits = sorted([p.fitness for p in players])
        best_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        med_fit = fits[len(fits)//2]
        mean = avg_fit
        std_fit = (sum((x - mean) ** 2 for x in fits) / max(1, (len(fits) - 1))) ** 0.5

        best_player = max(players, key=lambda p: p.fitness)

        best_pipes = getattr(best_player, "pipes_passed", 0)
        avg_pipes = sum(getattr(p, "pipes_passed", 0) for p in players) / len(players)
        best_fr = flap_rate(best_player)
        avg_fr = sum(flap_rate(p) for p in players) / len(players)
        best_ge = gap_err(best_player)
        avg_ge = sum(gap_err(p) for p in players) / len(players)

        div = diversity_stats(players)
        elapsed = time.perf_counter() - t0

        frac_capped = 1.0 if best_fit >= float(args.max_steps) else 0.0

        
        test_mean_fit = ""
        test_std_fit = ""
        test_mean_pipes = ""
        test_std_pipes = ""

        if args.eval_every and eval_seeds and (g % args.eval_every == 0):
            best_ws = genome_vector(best_player)
            res = evaluate_best_on_seeds(best_ws, eval_seeds, 
                                         max_steps=args.max_steps,
                                           threshold=args.threshold)
            test_mean_fit = res["test_mean_fitness"]
            test_std_fit = res["test_std_fitness"]
            test_mean_pipes = res["test_mean_pipes"]
            test_std_pipes = res["test_std_pipes"]
            last_outputs_for_hist = res["outputs"][-20000:]  

            
            np = try_np()
            if np is not None:
                np.savez_compressed(logdir / f"eval_outputs_gen{g:04d}.npz", outputs=np.array(res["outputs"], dtype=float))

        with gen_stats_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                g,
                best_fit, avg_fit, med_fit, std_fit,
                best_pipes, avg_pipes,
                best_fr, avg_fr,
                best_ge, avg_ge,
                div["w_dim"], div["w_var_mean"], div["w_std_mean"],
                frac_capped,
                test_mean_fit, test_std_fit,
                test_mean_pipes, test_std_pipes,
                elapsed
            ])

        if args.save_pop_every and (g % args.save_pop_every == 0):
            maybe_save_population_snapshot(logdir, g, players)

        if args.save_policy_every and (g % args.save_policy_every == 0):
            maybe_save_policy_snapshot(logdir, g, best_player, grid_n=args.policy_grid)

        players, best = evolve(players, pop_size=args.pop)
        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = best

        print(f"gen={g:4d} best={best_fit:6.1f} avg={avg_fit:6.1f} elapsed={elapsed:6.2f}s")

    if best_ever is not None:
        write_best_weights_csv(logdir / "best_genome_weights.csv", best_ever)

    
    if args.threshold_sweep:
        ts = args.threshold_sweep.replace(",", " ").split()
        thresholds = []
        for t in ts:
            try:
                thresholds.append(float(t))
            except Exception:
                pass

        if thresholds and best_ever is not None and eval_seeds:
            best_ws = genome_vector(best_ever)
            sweep_path = logdir / "threshold_sweep.csv"
            with sweep_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["threshold", "test_mean_fitness", "test_std_fitness"])
                for thr in thresholds:
                    res = evaluate_best_on_seeds(best_ws, eval_seeds, max_steps=args.max_steps, threshold=thr)
                    w.writerow([thr, res["test_mean_fitness"], res["test_std_fitness"]])

    pygame.quit()
    print(f"\nWrote logs to: {logdir.resolve()}")


if __name__ == "__main__":
    main()
