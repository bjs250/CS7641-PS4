import matplotlib.pyplot as plt
import numpy as np

def get_average_delta(deltas):
    max_iteration = 0
    for delta in deltas:
        if len(delta) > max_iteration:
            max_iteration = len(delta)


    delta_array = np.zeros((len(deltas), max_iteration))
    for i in range(len(deltas)):
        delta_array[i] = np.pad(np.asarray(deltas[i]), (0, max_iteration - len(deltas[i])))

    avg = np.mean(delta_array, 0)
    std = np.std(delta_array, 0)
    return avg, std


def plot_vi_convergence_size(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        average_delta, std_delta = get_average_delta(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Value Iteration, Frozen Lake (10 seeds), p=0.8")
    plt.xlabel("Iteration")
    plt.ylabel("Average Delta")
    plt.xlim(0, 1400)
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_vi_convergence_p(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for p, values in deltas.items():
        average_delta, std_delta = get_average_delta(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"p:{p}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Value Iteration, Frozen Lake (10 seeds), size=10")
    plt.xlabel("Iteration")
    plt.ylabel("Average Delta")
    plt.xlim(0, 300)
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_vi_convergence_d(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for d, values in deltas.items():
        average_delta, std_delta = get_average_delta(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"discount:{d}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Value Iteration, Frozen Lake (10 seeds), size=10, p=0.8")
    plt.xlabel("Iteration")
    plt.ylabel("Average Delta")
    plt.xlim(0, 200)
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_vi_convergence_dr(rewards: dict):

    X = rewards.keys()
    Y = [np.mean(rewards[discount]) for discount in rewards.keys()]
    err = [np.std(rewards[discount]) for discount in rewards.keys()]


    plt.plot(X, Y, 'bo')
    # plt.errorbar(X, Y, yerr=err)


    plt.title(f"Value Iteration, Frozen Lake (10 seeds), size=10, p=0.8")
    plt.xlabel("Discount Factor")
    plt.ylabel("Average Total Reward")
    plt.grid(b=True)
    # plt.legend()
    plt.show()

def plot_pi_convergence_size(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for size, values in deltas.items():
        average_delta, std_delta = get_average_delta(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Policy Iteration, Frozen Lake (10 seeds), p=0.8")
    plt.xlabel("Iteration")
    plt.ylabel("Average Delta")
    plt.xlim(0, 150)
    plt.ylim(0, 100)
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_pi_convergence_d(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    for d, values in deltas.items():
        average_delta, std_delta = get_average_delta(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"discount:{d}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Policy Iteration, Frozen Lake (10 seeds), size=10, p=0.8")
    plt.xlabel("Iteration")
    plt.ylabel("Average Delta")
    plt.xlim(0, 50)
    plt.ylim(0, 70 )
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_pi_convergence_dr(rewards: dict):

    X = rewards.keys()
    Y = [np.mean(rewards[discount]) for discount in rewards.keys()]
    err = [np.std(rewards[discount]) for discount in rewards.keys()]


    plt.plot(X, Y, 'bo')
    # plt.errorbar(X, Y, yerr=err)


    plt.title(f"Policy Iteration, Frozen Lake (10 seeds), size=10, p=0.8")
    plt.xlabel("Discount Factor")
    plt.ylabel("Average Total Reward")
    plt.grid(b=True)
    # plt.legend()
    plt.show()
