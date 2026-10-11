# A Reproducible Framework for Fair Evaluation of Variance-Aware Bandit Algorithms

This repository contains the implementation of various multi-armed bandit algorithms and a dashboard for visualizing their performance. The goal is to compare the effectiveness of different algorithms in maximizing rewards and minimizing regret over time.

#### Version Information

This project uses the branch `v1.0-SKILL2025` for the official SKILL 2025 submission. This documentation and code refers to the corresponding branch of this repository.

## Algorithms and Tuning Parameters

The following algorithms are implemented, each with its own set of tuning parameters:

- **ETC (Explore-then-Commit)**: Explores all available arms for a certain number of rounds before committing to the arm with the highest estimated reward.
   - Tuning Parameters: `exploration_rounds`
   - Scenarios:
      - `exploration_rounds`: 10
      - `exploration_rounds`: 100
      - `exploration_rounds`: 1000
      - `exploration_rounds`: 10000
      - `exploration_rounds`: 100000

- **Epsilon-Greedy**: Balances exploration and exploitation by choosing a random action with probability epsilon and the action with the highest estimated reward with probability \(1 - \epsilon\).
   - Tuning Parameters: `epsilon`
   - Scenarios:
      - `epsilon`: 0.5
      - `epsilon`: 0.1
      - `epsilon`: 0.01
      - `epsilon`: 0.05
      - `epsilon`: 0.005

- **UCB (Upper Confidence Bound)**: Selects the arm with the highest upper confidence bound to balance exploration and exploitation.
   - Tuning Parameters: None

- **UCB-Tuned**: Adjusts the confidence bound by considering the variance of the rewards.
   - Tuning Parameters: None

- **UCB-V**: Incorporates variance estimates into the upper confidence bounds.
   - Tuning Parameters: `theta`, `c`, `b`
   - Scenarios:
      - `theta`: 1, `c`: 1, `b`: 1

- **PAC-UCB**: Guarantees with high probability that the regret is close to the optimal policy.
   - Tuning Parameters: `c`, `b`, `q`, `beta`
   - Scenarios:
      - `c`: 1, `b`: 1, `q`: 1.3, `beta`: 0.05

- **UCB-Improved**: Enhances UCB with more sophisticated exploration strategies.
   - Tuning Parameters: `delta`
   - Scenarios:
      - `delta`: 1

- **EUCBV (Efficient-UCB with Variance)**: Uses empirical estimates of variance to adjust the upper confidence bounds.
   - Tuning Parameters: `rho`
   - Scenarios:
      - `rho`: 0.5


## Bandit Model

The bandit model used in this repository focuses on a multi-armed bandit problem with Bernoulli-distributed arms. The arms are set with the reward probabilities for each arm of $[0.8, 0.89, 0.895, 0.9]$ and can be chosen for two- or three-armed scenarios as well as all possible permutations

Each algorithm is run for 100 rounds, and the results are stored in separate directories for different time steps. Additionally, there is a 'results_average' file for each algorithm, providing the average values for each time step based on 100 samples.

## Plots in the Dashboard

Visualizations and dashboards were created using Plotly and Dash. The dashboard shows the three experiments from the study:

- **Scenario A (Baseline):** p1 = 0.8, p2 = 0.9
- **Scenario B (Low-Variance Micro-Gap):** p1 = 0.895, p2 = 0.9
- **Scenario C (High-Variance Micro-Gap):** p1 = 0.89, p2 = 0.895

For each scenario it shows three line charts over time for all 16 algorithm configurations, followed by a table of the values at T = 1,000,000:

1. **Average Regret**: mean total regret across the 100 trials.
2. **Reward Variance**: sample variance of the total reward across the 100 trials.
3. **Subopt. Ratio**: average number of suboptimal arm pulls divided by the timestep.

See [docs/workflow.md](docs/workflow.md) for the full workflow from simulation to dashboard.

## Setup Instructions

## 1. Clone the Repository

First, you need to clone the repository from GitHub to your local machine. Open your terminal (or command prompt) and run the following command:

```bash
git clone https://github.com/eelisee/bandit_playground.git
cd bandit_playground
```

Once the repository is cloned, checkout the Version 1.0 branch by running:
```bash
git checkout -b v1.0-SKILL2025 origin/v1.0-SKILL2025
```

## 2. Set Up a Virtual Environment

It is strongly recommended to use a virtual environment to manage the project's dependencies. You can create and activate a virtual environment by running the following commands:

__For macOS/Linux:__

```bash
python3 -m venv .venv
source .venv/bin/activate
```

__For Windows:__
```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

Once the virtual environment is activated, you need to install the required Python packages. Install them by running the following command:

```bash
pip install -r requirements.txt
```

This command will install all the dependencies listed in the ```requirements.txt``` file. Ensure that all packages install without errors.

## 4. Running the Scenario Simulations (optional)

The results for all three scenarios are already stored in `data/`, so you can skip this step and go straight to the dashboard. To regenerate a scenario, run from the repository root:

```bash
python src/run_scenario.py 0.8 0.9      # Scenario A
python src/run_scenario.py 0.895 0.9    # Scenario B
python src/run_scenario.py 0.89 0.895   # Scenario C
```

Each command runs all 16 algorithm configurations (100 trials, T = 1,000,000, seed 42) and overwrites that scenario's results in `data/`. One scenario takes about 15–20 minutes on a 12-core machine. Run the scenarios one after another, not in parallel.

## 5. Running the Dashboard

Once the installation is complete, you can start the dashboard by running the following command:

```bash
python src/dashboard.py
```

After the command runs, your default web browser should automatically open with the dashboard at this URL:

```bash
http://127.0.0.1:8050
```

If it doesn't open automatically, you can manually copy and paste this URL into your browser.

The dashboard reads `data/` only at startup. After running a simulation, stop the dashboard (Ctrl + C) and start it again to see the new results.


## Reproducibility

The code is designed for easy extensibility:

- **Configuration Management**: All simulation parameters and settings are centrally defined in the configuration file ```config.py```, enabling quick adjustments without changing the core code.
- **Adding New Algorithms**: New multi-armed bandit algorithms can be integrated easily following a clear structure.
- **Customizing Plots**: Additional plots for analysis and visualization can be added with minimal changes.
- **Flexible Scenario Definition**: Different simulation scenarios can be defined via configurable arm distributions.

Detailed instructions for extending the simulation are available in the following documentation files:

- [Instructions to change configuration](docs/instructions_to_change_configuration.md)
- [Instructions to add new algorithms](docs/instructions_to_add_new_algorithms.md)
- [Instructions to add new plots](docs/instructions_to_add_new_plot.md)

## Authors

**Elise Wolf** – [elise.marie.wolf@students.uni-mannheim.de] - University of Mannheim Affiliation

## Citation

If you use this project, please cite
@misc{wolf2025frameworkfairevaluationvarianceaware,
      title={A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms}, 
      author={Elise Wolf},
      year={2025},
      eprint={2510.27001},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.27001}, 
}

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

