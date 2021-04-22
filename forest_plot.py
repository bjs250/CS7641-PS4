import matplotlib.pyplot as plt

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