

![Preview](/preview.gif)


This solo project trains a Flappy Bird agent using a simple neuroevolution approach, built to run fast enough to produce clean learning curves and statistics. I originally considered using the full NEAT algorithm, but I chose to start from scratch so I could understand inheritance, mutation, and selection more directly. The game began as a pygame implementation, which is great for visualization but too slow for running many generations, so I created a headless trainer (fast_train_plus) that keeps the same physics and collision logic without rendering. Each bird uses a minimal perceptron-style policy: 2–3 vision inputs (distance-based features) plus a bias input fixed at 1, producing a single output. The bird flaps when the output crosses a threshold, which can be viewed as a simple neuron “firing.” Training runs by simulating a population, scoring fitness (lifespan / pipes), selecting top performers, and cloning + mutating weights to form the next generation. To measure stability instead of a single lucky run, I also evaluate the best policy on fixed seeds (eval_seeds 0–4) across generations. The final deliverables include learning curves (best/avg, mean±std bands, fixed-seed evaluation) with zoomed-in views (1–10, 1–30, 1–40, 1–100, 1–200) and a modified main.py demo that overlays key stats in real time.

Run configuration used for the main results in this paper:

python fast_train_plus.py --seed 0 --gens 200 --pop 20 --max_steps 4000 --logdir runs/exp1 --save_pop_every 1 --save_policy_every 1 --eval_every 1 --eval_seeds 0,1,2,3,4

View Champion

python viewer_player.py --weights_csv runs/exp1/best_genome_weights.csv
