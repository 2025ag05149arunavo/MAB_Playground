"""
Runs all algorithm configurations from config.py for one two-armed scenario
and writes the results to a separate output directory (data/ is not touched).

Usage (from the repository root):
    python src/run_scenario.py <p1> <p2> <output_dir>

Example:
    python src/run_scenario.py 0.8 0.9 runs/scenario_A
"""
import os
import sys
import csv
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from config import algorithm_strategy_pairs, time_horizons, global_seed
from utils.simulation_utils import general_simulation
from utils.bandit_algorithm import BanditAlgorithm

def run(args):
    index, arm_means, out_dir = args
    algo, strategy = algorithm_strategy_pairs[index]
    algorithm = BanditAlgorithm(algo.name)
    start = time.time()
    general_simulation(algorithm, arm_means, time_horizons, strategy["strategy_fn"], rng=np.random.default_rng(global_seed), **strategy["params"])

    algorithm.save_results_to_csv(os.path.join(out_dir, f"{index}_results.csv"))
    with open(os.path.join(out_dir, f"{index}_average.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Timestep', 'Average Total Reward', 'Average Suboptimal Arms', 'Average Regret', 'Average Zeros Count', 'Average Ones Count'])
        writer.writerows(algorithm.calculate_average_results())
    print(f"Done {algo.name} {strategy['params']} in {time.time() - start:.0f}s", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    arm_means = (float(sys.argv[1]), float(sys.argv[2]))
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    max_workers = max(1, int(os.cpu_count() * 0.75))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run, [(i, arm_means, out_dir) for i in range(len(algorithm_strategy_pairs))]))
