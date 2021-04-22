import matplotlib.pyplot as plt
import numpy as np

def plot_vi_forest_convergence(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        X = range(len(values))
        Y = values
        plt.plot(X, Y, color=colors[i], label=f"p:{size}")
        # plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Value Iteration, Forest, size=10, r1=1, r2=5")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_pi_forest_convergence_size(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        X = range(len(values))
        Y = values
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        # plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Policy Iteration, Forest, p=0.1, r1=1, r2=5")
    plt.xlabel("Iteration")
    plt.ylabel("Delta")
    plt.xticks([1,2,3,4])
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_pi_forest_convergence_p(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        X = range(len(values))
        Y = values
        plt.plot(X, Y, color=colors[i], label=f"p:{size}")
        # plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Policy Iteration, Forest, size=10, r1=1, r2=5")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(b=True)
    plt.legend()
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_ql_forest_convergence_size(deltas:dict):
    colors = ['b', 'g', 'y', 'r']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        X = range(len(values))
        Y = values
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        i += 1

    plt.title(f"Q-learning, Forest, size=10, r1=1, r2=5")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(b=True)
    plt.xlim([0, 500])
    plt.ylim([0, .100])
    plt.legend()
    plt.show()

def plot_ql_forest_reward_size(rewards:dict, epsilon:list, w):
    colors = ['b', 'g', 'y', 'r']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in rewards.items():
        X = range(w-1, len(values))
        Y = moving_average(values, w)
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        i += 1

    X = range(len(values))
    Y = epsilon[0:len(values)]
    plt.plot(X, Y, label="epsilon")

    plt.title(f"Q-learning, Forest, size=10, r1=1, r2=5")
    plt.xlabel("Iteration")
    plt.ylabel(f"Mean Reward (w={w})")
    plt.grid(b=True)
    plt.xlim([0, 1500])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.show()