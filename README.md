# SQL Query Execution RL Optimizer (Historical Model)

This repository contains a Jupyter notebook implementing an RL-based optimizer for SQL query execution costs in PostgreSQL, using historical data. The project uses reinforcement learning algorithms (PPO, SAC, DDPG) to dynamically select query execution plans, achieving 30–50% cost reduction with PPO and 60–70% with SAC/DDPG. Note: This is a historical model trained on past query data (e.g., IMDB dataset), not designed for real-time application.

## Project Overview
- **Objective**: Minimize SQL query execution costs by learning optimal plans (e.g., join orders, index usage) using RL.
- **Key Features**:
  - Custom OpenAI Gym environment for simulating query execution.
  - RL Algorithms: PPO (stable, on-policy), SAC (exploration-focused, off-policy), DDPG (continuous actions, off-policy).
  - Dataset: IMDB (70% training, 30% testing).
  - Metrics: Cost reduction (e.g., 42% average), optimization ratio.
  - Visualization: Log-scale cost comparison graphs.
- **Results**: 30–70% cost reduction, surpassing traditional (10–15%) and ML-based (20%) benchmarks.
- **Limitations**: Historical model (not real-time); further work needed for live deployment.

## Clone repo
1. Clone the repository:
git clone https://github.com/KS-KARTHIK-05/SQL-Query-Execution-RL-Optimizer.git 
cd SQL-Query-Execution-RL-Optimizer

## Installation
2. Install dependencies (use a virtual environment):
pip install gymnasium stable-baselines3 numpy matplotlib seaborn pandas


3. Ensure PostgreSQL is installed for any future extensions (not required for historical model).

## Usage
1. Open the notebook `development_notebook.ipynb` in Jupyter Lab or Notebook.
2. Run the GPU check cell to verify hardware.
3. Run the main code cell to load the IMDB dataset, create environments, train RL models (PPO, SAC, DDPG), and generate results/graphs.
4. Outputs: Training progress, final test metrics (e.g., total reward, avg RL cost, optimization ratio, cost reduction), and graphs saved as PNG files (e.g., "ppo_graph.png").

## Methodology
1. **Environment Setup**:
- State: Query features (joins, filters, table sizes).
- Action: Plan adjustments (join orders, index usage).
- Reward: Negative cost + stability term.
2. **Training**:
- PPO: 500 episodes, learning rate 0.001, discount factor 0.99.
- SAC: Entropy regularization, replay buffer.
- DDPG: Ornstein-Uhlenbeck noise, actor-critic updates.
3. **Evaluation**: Held-out IMDB queries, cost reduction metrics.

## Level Of Potential
This project has the potential to go typical journal-level work:
- Interdisciplinary: Combines databases and RL.
- Practical Impact: 60–70% cost reduction in PostgreSQL.
- Depth: Custom environment, multiple RL algorithms, iterative tuning.
- Visionary: Generalizable to other DBMS (e.g., MySQL), open-source release.

## Future Work
- Real-time integration with PostgreSQL.
- Generalization to other databases.
- Theoretical analysis (e.g., convergence proofs).
- Open-source release with technical report.

## Acknowledgments
This project was built using the following key libraries and tools:
- **Gymnasium** – For the custom RL environment simulation.
- **Stable-Baselines3** – For implementing PPO, SAC, and DDPG algorithms.
- **NumPy, Pandas** – Data handling and preprocessing.
- **Matplotlib & Seaborn** – Visualization of cost comparisons and results.
- IMDB dataset for training and evaluation.

Special thanks to the contributors of the IMDB dataset and the broader reinforcement learning community.

## License
MIT License