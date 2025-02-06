#!/usr/bin/env python3
"""

This script simulates a nonstationary 10–armed bandit problem with four algorithms:
  1. Sample–Average (ε–greedy with decreasing step size)
  2. Constant Step–Size (ε–greedy with fixed α)
  3. Gradient Bandit (preference–based with softmax and average–reward baseline)
  4. Thompson Sampling (using a Gaussian sampling approximation)

At each time step, the simulation computes:
  - Reward obtained.
  - Whether the optimal action was chosen.
  - Instantaneous regret = (max(true values) – reward).

Across many runs the simulation aggregates these into:
  - Average Reward
  - % Optimal Action
  - Cumulative Reward
  - Average Instantaneous Regret
  - Reward Variance

"""

import argparse
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random, lax, jit, vmap
from functools import partial
import numpy as np
from matplotlib.animation import PillowWriter




# ------------------------------
# Global Constants (can be tunned futher)
# ------------------------------
NUM_ARMS = 10
EPSILON = 0.1         # for epsilon-greedy methods
Q_WALK_STD = 0.01     # std dev for random walk on true values
CONST_ALPHA = 0.1     # constant step-size for constant-step agent
GRAD_ALPHA = 0.1      # step-size for gradient bandit updates
THOMPSON_LARGE_STD = 1e3  # large std for untried arms in Thompson Sampling

# ------------------------------
# Utility Functions
# ------------------------------
@jit
def softmax(x):
    max_val = jnp.max(x)
    exp_x = jnp.exp(x - max_val)
    return exp_x / jnp.sum(exp_x)

# ------------------------------
# Per–Step Simulation Functions (each returns: reward, optimal_flag, inst_regret)
# ------------------------------

@jit
def sample_average_step(state, _):
    # state: (q_true, Q, counts, key)
    q_true, Q, counts, key = state

    # Update true values (random walk)
    key, subkey = random.split(key)
    q_walk = random.normal(subkey, shape=(NUM_ARMS,)) * Q_WALK_STD
    q_true = q_true + q_walk

    optimal_action = jnp.argmax(q_true)
    optimal_reward = jnp.max(q_true)

    key, subkey = random.split(key)
    rand_val = random.uniform(subkey)
    action = jnp.where(rand_val < EPSILON,
                       random.randint(key, shape=(), minval=0, maxval=NUM_ARMS),
                       jnp.argmax(Q))

    key, subkey = random.split(key)
    reward = random.normal(subkey) + q_true[action]
    inst_regret = optimal_reward - reward

    new_count = counts[action] + 1.0
    counts = counts.at[action].set(new_count)
    alpha = 1.0 / new_count
    Q = Q.at[action].set(Q[action] + alpha * (reward - Q[action]))

    optimal_flag = jnp.where(action == optimal_action, 1.0, 0.0)
    new_state = (q_true, Q, counts, key)
    return new_state, (reward, optimal_flag, inst_regret)


@jit
def constant_step_step(state, _):
    # state: (q_true, Q, key)
    q_true, Q, key = state

    key, subkey = random.split(key)
    q_walk = random.normal(subkey, shape=(NUM_ARMS,)) * Q_WALK_STD
    q_true = q_true + q_walk

    optimal_action = jnp.argmax(q_true)
    optimal_reward = jnp.max(q_true)

    key, subkey = random.split(key)
    rand_val = random.uniform(subkey)
    action = jnp.where(rand_val < EPSILON,
                       random.randint(key, shape=(), minval=0, maxval=NUM_ARMS),
                       jnp.argmax(Q))

    key, subkey = random.split(key)
    reward = random.normal(subkey) + q_true[action]
    inst_regret = optimal_reward - reward

    Q = Q.at[action].set(Q[action] + CONST_ALPHA * (reward - Q[action]))
    optimal_flag = jnp.where(action == optimal_action, 1.0, 0.0)
    new_state = (q_true, Q, key)
    return new_state, (reward, optimal_flag, inst_regret)


@jit
def gradient_bandit_step(state, _):
    # state: (q_true, H, avg_reward, t, key)
    q_true, H, avg_reward, t, key = state

    key, subkey = random.split(key)
    q_walk = random.normal(subkey, shape=(NUM_ARMS,)) * Q_WALK_STD
    q_true = q_true + q_walk

    optimal_action = jnp.argmax(q_true)
    optimal_reward = jnp.max(q_true)

    probs = softmax(H)
    key, subkey = random.split(key)
    r = random.uniform(subkey)
    cum_probs = jnp.cumsum(probs)
    action = jnp.argmax(cum_probs >= r)

    key, subkey = random.split(key)
    reward = random.normal(subkey) + q_true[action]
    inst_regret = optimal_reward - reward

    t_new = t + 1.0
    avg_reward = avg_reward + (reward - avg_reward) / t_new

    def update_pref(a, H_val):
        indicator = jnp.where(a == action, 1.0, 0.0)
        return H_val + GRAD_ALPHA * (reward - avg_reward) * (indicator - probs[a])
    H = jnp.array([update_pref(a, H[a]) for a in range(NUM_ARMS)])

    optimal_flag = jnp.where(action == optimal_action, 1.0, 0.0)
    new_state = (q_true, H, avg_reward, t_new, key)
    return new_state, (reward, optimal_flag, inst_regret)


@jit
def thompson_sampling_step(state, _):
    # state: (q_true, Q, counts, key)
    q_true, Q, counts, key = state

    key, subkey = random.split(key)
    q_walk = random.normal(subkey, shape=(NUM_ARMS,)) * Q_WALK_STD
    q_true = q_true + q_walk

    optimal_action = jnp.argmax(q_true)
    optimal_reward = jnp.max(q_true)

    def sample_arm(a):
        std = jnp.where(counts[a] > 0, 1.0 / jnp.sqrt(counts[a]), THOMPSON_LARGE_STD)
        key_local = random.fold_in(key, a)
        return random.normal(key_local) * std + Q[a]
    samples = jnp.array([sample_arm(a) for a in range(NUM_ARMS)])
    action = jnp.argmax(samples)

    key, subkey = random.split(key)
    reward = random.normal(subkey) + q_true[action]
    inst_regret = optimal_reward - reward

    new_count = counts[action] + 1.0
    counts = counts.at[action].set(new_count)
    alpha = 1.0 / new_count
    Q = Q.at[action].set(Q[action] + alpha * (reward - Q[action]))

    optimal_flag = jnp.where(action == optimal_action, 1.0, 0.0)
    new_state = (q_true, Q, counts, key)
    return new_state, (reward, optimal_flag, inst_regret)

# ------------------------------
# Simulation Run Functions (using lax.scan)
# ------------------------------
@partial(jit, static_argnums=(1,))
def run_sample_average(key, num_steps):
    init_state = (jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), key)
    state, outputs = lax.scan(sample_average_step, init_state, jnp.arange(num_steps))
    rewards, optimal_flags, regrets = outputs
    return rewards, optimal_flags, regrets

@partial(jit, static_argnums=(1,))
def run_constant_step(key, num_steps):
    init_state = (jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), key)
    state, outputs = lax.scan(constant_step_step, init_state, jnp.arange(num_steps))
    rewards, optimal_flags, regrets = outputs
    return rewards, optimal_flags, regrets

@partial(jit, static_argnums=(1,))
def run_gradient_bandit(key, num_steps):
    init_state = (jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), 0.0, 0.0, key)
    state, outputs = lax.scan(gradient_bandit_step, init_state, jnp.arange(num_steps))
    rewards, optimal_flags, regrets = outputs
    return rewards, optimal_flags, regrets

@partial(jit, static_argnums=(1,))
def run_thompson_sampling(key, num_steps):
    init_state = (jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), jnp.zeros(NUM_ARMS), key)
    state, outputs = lax.scan(thompson_sampling_step, init_state, jnp.arange(num_steps))
    rewards, optimal_flags, regrets = outputs
    return rewards, optimal_flags, regrets

def run_simulations(algorithm: str, num_runs: int, num_steps: int, seed: int = 0):
    """
    Run many simulation runs in parallel using vmap.
    Returns:
      avg_rewards: average reward per time step (length num_steps)
      opt_perc: % optimal actions per time step (length num_steps)
      avg_regret: average instantaneous regret per step (length num_steps)
      reward_var: variance of rewards per step (length num_steps)
    """
    base_key = random.PRNGKey(seed)
    keys = random.split(base_key, num_runs)
    
    if algorithm == "sample_average":
        sim_run = vmap(run_sample_average, in_axes=(0, None))
    elif algorithm == "constant":
        sim_run = vmap(run_constant_step, in_axes=(0, None))
    elif algorithm == "gradient":
        sim_run = vmap(run_gradient_bandit, in_axes=(0, None))
    elif algorithm == "thompson":
        sim_run = vmap(run_thompson_sampling, in_axes=(0, None))
    else:
        raise ValueError("Unknown algorithm")
    
    rewards_all, optimal_all, regret_all = sim_run(keys, num_steps)
    avg_rewards = jnp.mean(rewards_all, axis=0)
    opt_perc = 100 * jnp.mean(optimal_all, axis=0)
    avg_regret = jnp.mean(regret_all, axis=0)
    reward_var = jnp.var(rewards_all, axis=0)
    return avg_rewards, opt_perc, avg_regret, reward_var

# ------------------------------
# Animation Helper Function
# ------------------------------
def animate_results(results, steps, metric="avg_rewards", interval=50, frame_step=100):
    """
    Animate one selected performance metric over time for all algorithms.
    
    Args:
      results: dict mapping algorithm name to a dict of metric arrays.
      steps: 1D array of step indices.
      metric: one of "avg_rewards", "opt_perc", "cum_rewards", "inst_regret", or "reward_var".
      interval: delay between frames (ms).
      frame_step: update every frame_step steps.
    Returns:
      The FuncAnimation object.
    """
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    lines = {}
    for alg in results:
        line, = ax.plot([], [], label=alg)
        lines[alg] = line
    ax.set_xlim(0, steps[-1])
    
    # For cumulative rewards, use the precomputed array; otherwise, use metric directly.
    if metric == "cum_rewards":
        all_data = np.concatenate([np.array(results[alg]["cum_rewards"]) for alg in results])
    else:
        all_data = np.concatenate([np.array(results[alg][metric]) for alg in results])
    ax.set_ylim(np.min(all_data), np.max(all_data))
    
    titles = {
        "avg_rewards": "Animated Average Reward vs. Steps",
        "opt_perc": "Animated % Optimal Action vs. Steps",
        "cum_rewards": "Animated Cumulative Reward vs. Steps",
        "inst_regret": "Animated Instantaneous Regret vs. Steps",
        "reward_var": "Animated Reward Variance vs. Steps"
    }
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(titles.get(metric, metric))
    ax.set_xlabel("Steps")
    ax.legend()

    def init():
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values())

    def update(frame):
        current_steps = steps[:frame]
        for alg in results:
            current_metric = results[alg][metric][:frame]
            lines[alg].set_data(current_steps, current_metric)
        return list(lines.values())

    num_frames = len(steps) // frame_step
    anim = FuncAnimation(fig, update, frames=range(frame_step, len(steps)+1, frame_step),
                         init_func=init, blit=True, interval=interval)
    return anim

# ------------------------------
# Main Function and Plotting
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Nonstationary Bandit Simulation using JAX (All Algorithms + Deep Metrics + Animations)")
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of time steps")
    parser.add_argument("--num_runs", type=int, default=2000, help="Number of runs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--animate_all", action="store_true", help="If set, generate animations for all metrics")
    parser.add_argument("--animate", action="store_true", help="If set, generate an animation for the chosen metric")
    parser.add_argument("--metric", type=str, default="avg_rewards",
                        choices=["avg_rewards", "opt_perc", "cum_rewards", "inst_regret", "reward_var"],
                        help="Which metric to animate (if --animate is used)")
    args = parser.parse_args()

    algorithms = ["sample_average", "constant", "gradient", "thompson"]
    results = {}
    for alg in algorithms:
        print(f"Simulating {alg}...")
        avg_rewards, opt_perc, avg_regret, reward_var = run_simulations(alg, args.num_runs, args.num_steps, args.seed)
        cum_rewards = np.cumsum(np.array(avg_rewards))
        results[alg] = {
            "avg_rewards": np.array(avg_rewards),
            "opt_perc": np.array(opt_perc),
            "cum_rewards": cum_rewards,
            "inst_regret": np.array(avg_regret),
            "reward_var": np.array(reward_var)
        }

    steps = np.arange(args.num_steps)
    # Static Plots
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    for alg in algorithms:
        plt.plot(steps, results[alg]["avg_rewards"], label=alg)
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs. Steps")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    for alg in algorithms:
        plt.plot(steps, results[alg]["opt_perc"], label=alg)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Optimal Action % vs. Steps")
    plt.legend()
    
    plt.subplot(2, 3, 3)
    for alg in algorithms:
        plt.plot(steps, results[alg]["cum_rewards"], label=alg)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs. Steps")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    for alg in algorithms:
        plt.plot(steps, results[alg]["inst_regret"], label=alg)
    plt.xlabel("Steps")
    plt.ylabel("Instantaneous Regret")
    plt.title("Instantaneous Regret vs. Steps")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    for alg in algorithms:
        plt.plot(steps, results[alg]["reward_var"], label=alg)
    plt.xlabel("Steps")
    plt.ylabel("Reward Variance")
    plt.title("Reward Variance vs. Steps")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    writer = PillowWriter(fps=10)
    # Animation(s)
    if args.animate_all:
        metrics = ["avg_rewards", "opt_perc", "cum_rewards", "inst_regret", "reward_var"]
        for m in metrics:
            print(f"Generating animation for {m}...")
            anim = animate_results(results, steps, metric=m, interval=50, frame_step=100)
            filename = f"animation_{m}.gif"
            anim.save(filename, writer=writer)
            print(f"Saved animation for {m} as {filename}")
    elif args.animate:
        anim = animate_results(results, steps, metric=args.metric, interval=50, frame_step=100)
        filename = f"animation_{args.metric}.gif"
        anim.save(filename, writer=writer)
        print(f"Saved animation for {args.metric} as {filename}")

if __name__ == "__main__":
    main()
