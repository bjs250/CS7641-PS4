import matplotlib.pyplot as plt
import numpy as np

###### CARTPOLE ######

def plot_cartpole_vi_reward(rewards: list, w):

    moving_average(rewards, w)

    X = range(w-1, len(rewards))
    Y = moving_average(rewards, w)

    plt.plot(X, Y, label="epsilon")

    plt.title(f"Value Iteration, Cartpole, w={w}")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.grid(b=True)
    plt.legend()
    plt.show()

def plot_cartpole_vi_deltas(deltas: dict):

    colors = ['b', 'g', 'y', 'r']
    facecolors = ['lightblue', 'lightgreen', 'khaki', 'tomato']
    alphas = [.6, .3, .2, .1]
    i = 0

    # colors = ['b']
    # facecolors = ['lightblue']
    # alphas = [.6]
    # i = 0

    for size, values in deltas.items():
        average_delta, std_delta = get_average_delta_cartpole(values)
        X = range(0, average_delta.size)
        Y = average_delta
        plt.plot(X, Y, color=colors[i], label=f"size:{size}")
        plt.fill_between(X, Y - std_delta, Y + std_delta, facecolor=facecolors[i], alpha=alphas[i])
        i += 1

    plt.title(f"Value Iteration, Cartpole (10 seeds)")
    plt.xlabel("Iteration")
    plt.ylabel("Delta")
    plt.xlim([0, 20])
    plt.ylim([0, 30])
    plt.xticks(range(0, 20, 1))
    plt.grid(b=True)
    plt.legend()
    plt.show()


    # X = range(0, len(deltas))
    # Y = deltas
    # plt.plot(X, Y)
    # plt.title(f"Value Iteration, Cartpole (10 seeds)")
    # plt.xlabel("Iteration")
    # plt.ylabel("Delta")
    # plt.grid(b=True)
    # plt.legend()
    # plt.show()

#################3

def get_average_delta_cartpole(runs:dict):
    max_iteration = 0
    for seed, values in runs.items():
        if len(values) > max_iteration:
            max_iteration = len(values)

    delta_array = np.zeros((len(runs.keys()), max_iteration))
    for i in range(len(runs.keys())):
        delta_array[i] = np.pad(np.asarray(runs[i]), (0, max_iteration - len(runs[i])))

    avg = np.mean(delta_array, 0)
    std = np.std(delta_array, 0)
    return avg, std



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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_q(rewards, w):

    agg_rewards = {}

    for seed, seed_dict in rewards[5].items():
        for tuple, values in seed_dict.items():
            final_value = moving_average(values, 1000)[-1]
            if tuple not in agg_rewards.keys():
                agg_rewards[tuple] = final_value
            else:
                agg_rewards[tuple] += final_value
            # print(seed, tuple, moving_average(values, 1000)[-1])

    norm_agg_rewards = {}
    for tuple, value in agg_rewards.items():
        norm_agg_rewards[tuple] = agg_rewards[tuple] / 10

    sorted_norm_agg_rewards = {k: v for k, v in sorted(norm_agg_rewards.items(), key=lambda item: item[1])}

    for k,v in sorted_norm_agg_rewards.items():
        print(k, v)
    # X = [tuple[0] for tuple, value in norm_agg_rewards.items()]
    # Y = [value for tuple, value in norm_agg_rewards.items()]
    # plt.plot(X, Y, 'bo')
    #
    #
    # plt.title(f"Q Learning, Frozen Lake (10 seeds), size=5, p=0.8, w=1000")
    # plt.xlabel("Episode")
    # plt.ylabel("Average Total Reward")
    # plt.grid(b=True)
    # # plt.legend()
    # plt.show()

def get_average_reward(rewards: dict, size):
    num_seeds = len(rewards[size].keys())
    avg = [sum(x) for x in zip(*rewards[size].values())]
    # avg = list()
    # for seed, values in rewards[size].items():
    #     if len(avg) == 0:
    #         avg = values
    #     else:
    #         avg += values
    return [x/num_seeds for x in avg]

def plot_episode_rewards(rewards: dict, epsilons, w):

    average_rewards = get_average_reward(rewards, 10)

    X = range(w-1, len(average_rewards))
    Y = moving_average(average_rewards, w)

    plt.plot(X, Y, label="reward")

    X = range(0, len(epsilons))
    Y = epsilons

    plt.plot(X, Y, label="epsilon")

    plt.title(f"Q Learning, Frozen Lake (10 seeds), size=10, p=0.99, w=1000")
    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # plt.ylim(0, 0.5)
    plt.grid(b=True)
    plt.legend()
    plt.show()