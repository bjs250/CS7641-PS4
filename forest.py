from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration
from hiive.mdptoolbox.example import forest

import forest_plot

# PROBLEM_SIZES = [.2, .1, .05, .01]
PROBLEM_SIZES = [10, 20, 30, 40]

def value_iteration():
    deltas = {}
    rewards = {}
    for size in PROBLEM_SIZES:
        P, R = forest(S=size, r1=1, r2=5, p=.1)
        vi = ValueIteration(P, R, 0.9, max_iter=10)
        vi.run()
        delta = [vi.run_stats[i]['Error'] for i in range(len(vi.run_stats))]
        reward = [vi.run_stats[i]['Reward'] for i in range(len(vi.run_stats))]
        deltas[size] = delta
        rewards[size] = reward
        print(vi.policy)
        print(vi.S)

def policy_iteration():
    deltas = {}
    rewards = {}
    for size in PROBLEM_SIZES:
        P, R = forest(S=size, r1=1, r2=5, p=.1)
        pi = PolicyIteration(P, R, 0.9, max_iter=10)
        pi.run()
        delta = [pi.run_stats[i]['Error'] for i in range(len(pi.run_stats))]
        reward = [pi.run_stats[i]['Reward'] for i in range(len(pi.run_stats))]
        deltas[size] = delta
        rewards[size] = reward
        print(pi.policy)
        print(pi.S)

    # forest_plot.plot_pi_forest_convergence_size(rewards)

    deltas = {}
    rewards = {}
    for p in [.2, .1, .05, .01]:
        P, R = forest(S=10, r1=1, r2=5, p=p)
        pi = PolicyIteration(P, R, 0.9, max_iter=10)
        pi.run()
        delta = [pi.run_stats[i]['Error'] for i in range(len(pi.run_stats))]
        reward = [pi.run_stats[i]['Reward'] for i in range(len(pi.run_stats))]
        deltas[p] = delta
        rewards[p] = reward
        print(pi.policy)
        print(pi.S)

    forest_plot.plot_pi_forest_convergence_p(rewards)


policy_iteration()
# forest_plot.plot_vi_forest_convergence(deltas)
# forest_plot.plot_vi_forest_convergence(rewards)
