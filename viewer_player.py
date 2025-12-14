"""
viewer_player.py
----------------
Pygame visualizer that replays the SAME headless environment used in fast_train_plus.py,
but draws it so you can debug what's going on.

Key goal: **no mismatch**.
- Uses Player() from your repo (same physics in Player.update / Player.bird_flap).
- Uses the same Pipe class + vision computation as fast_train_plus.py.
- Loads champion weights from best_genome_weights.csv (the file your trainer writes).

Run:
  python viewer_player.py --weights_csv runs/exp1/best_genome_weights.csv --seed 0 --threshold 0.73

Controls:
  [ / ]   decrease/increase ticks-per-frame (speed up without changing physics)
  - / =   decrease/increase FPS cap (rendering speed)
  R       reset episode
  ESC     quit
"""

import argparse
import csv
import os
import random
import time

import pygame

import config
from player import Player

# ---- match fast_train_plus defaults (only pipe geometry / world size lives here) ----
SCREEN_W = 500
SCREEN_H = 500
GROUND_Y = 500
BIRD_X = 50

PIPE_W = 15
PIPE_OPENING = 110
PIPE_SPEED = 2.5
SPAWN_X = SCREEN_W

DEFAULT_CEILING_Y = 30  # IMPORTANT: without this, extreme flap velocities can "escape" above the pipes.


class Pipe:
    """Single pipe pair, keeps pygame.Rects so Player.pipe_collision() works."""
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
    # Same math as your Player.look(), but without drawing.
    player.vision[0] = max(0, player.rect.center[1] - pipe.top_rect.bottom) / SCREEN_H
    player.vision[1] = max(0, pipe.x - player.rect.center[0]) / SCREEN_W
    player.vision[2] = max(0, pipe.bottom_rect.top - player.rect.center[1]) / SCREEN_H


def load_weights_csv(path: str):
    weights = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            weights.append(float(row["weight"]))
    return weights


def set_player_weights(pl: Player, ws):
    # Order must match Brain.__init__ creation: inputs 0..2 then bias node (id 3).
    i = 0
    for c in getattr(pl.brain, "connections", []):
        if hasattr(c, "weight"):
            c.weight = float(ws[i])
        elif hasattr(c, "w"):
            c.w = float(ws[i])
        i += 1
        if i >= len(ws):
            break


def draw_text(win, font, x, y, s):
    img = font.render(s, True, (20, 20, 20))
    win.blit(img, (x, y))


def reset_episode(seed: int, weights, ceiling_y: int):
    rng = random.Random(int(seed))
    config.pipes = [Pipe(SPAWN_X, rng)]
    config.window = None

    pl = Player()
    pl.rect.x = BIRD_X
    pl.rect.y = 200
    pl.vel = 0
    pl.flap = False
    pl.alive = True
    pl.lifespan = 0

    set_player_weights(pl, weights)
    pl.brain.generate_net()

    ground_rect = pygame.Rect(0, GROUND_Y, SCREEN_W, 5)
    return rng, pl, ground_rect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.73)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--ticks_per_frame", type=int, default=1, help="How many physics ticks to run per render frame.")
    parser.add_argument("--ceiling_y", type=int, default=DEFAULT_CEILING_Y, help="Kill bird if rect.y < this value.")
    args = parser.parse_args()

    weights = load_weights_csv(args.weights_csv)

    pygame.init()
    win = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Flappy Bird — Champion Viewer (Player-based)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    rng, pl, ground_rect = reset_episode(args.seed, weights, args.ceiling_y)

    ticks_per_frame = max(1, int(args.ticks_per_frame))
    fps_cap = max(1, int(args.fps))

    last_reset_t = time.time()
    pipes_passed = 0

    running = True
    while running:
        clock.tick(fps_cap)

        # ---- events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    rng, pl, ground_rect = reset_episode(args.seed, weights, args.ceiling_y)
                    pipes_passed = 0
                    last_reset_t = time.time()
                elif event.key == pygame.K_LEFTBRACKET:  # [
                    ticks_per_frame = max(1, ticks_per_frame - 1)
                elif event.key == pygame.K_RIGHTBRACKET:  # ]
                    ticks_per_frame = min(200, ticks_per_frame + 1)
                elif event.key == pygame.K_MINUS:
                    fps_cap = max(5, fps_cap - 5)
                elif event.key == pygame.K_EQUALS:
                    fps_cap = min(240, fps_cap + 5)

        # ---- sim ticks ----
        for _ in range(ticks_per_frame):
            if not pl.alive:
                # short pause then reset
                if time.time() - last_reset_t > 0.5:
                    rng, pl, ground_rect = reset_episode(args.seed, weights, args.ceiling_y)
                    pipes_passed = 0
                    last_reset_t = time.time()
                break

            pipe = closest_pipe()
            prev_passed = pipe.passed
            pipe.step()

            if (not prev_passed) and pipe.passed:
                pipes_passed += 1

            if pipe.x + PIPE_W < 0:
                config.pipes[0].respawn()
                pipe = config.pipes[0]

            compute_vision(pl, pipe)

            out = pl.brain.feed_forward(pl.vision)
            pl.decision = float(out)

            # SAME threshold logic as your Player.think()
            if pl.decision > args.threshold:
                pl.bird_flap()

            pl.update(ground_rect)

            # IMPORTANT: add a ceiling "death" so huge flap velocities can't cheat by going off-screen.
            if pl.rect.y < args.ceiling_y:
                pl.alive = False

        # ---- draw ----
        win.fill((245, 245, 245))

        # pipes
        for p in config.pipes:
            pygame.draw.rect(win, (60, 180, 80), p.top_rect)
            pygame.draw.rect(win, (60, 180, 80), p.bottom_rect)

        # ground line
        pygame.draw.rect(win, (30, 30, 30), ground_rect)

        # bird
        pygame.draw.rect(win, (80, 120, 240), pl.rect)

        # HUD
        draw_text(win, font, 12, 10, f"seed={args.seed}  thr={args.threshold:.2f}")
        draw_text(win, font, 12, 30, f"ticks/frame={ticks_per_frame}  fps_cap={fps_cap}")
        draw_text(win, font, 12, 50, f"alive={pl.alive}  y={pl.rect.y}  vel={pl.vel:.2f}")
        draw_text(win, font, 12, 70, f"output={getattr(pl, 'decision', 0.0):.3f}")
        draw_text(win, font, 12, 90, f"pipes_passed={pipes_passed}")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # prevents the pygame "Hello from the pygame community" spam on some setups
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
    main()
