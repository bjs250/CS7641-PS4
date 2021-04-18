import gym
import numpy as np
import time
import pickle

import value_iteration
import policy_iteration
import q_learning
import plotting

SMALL_SIZE = 5
MEDIUM_SIZE = 10
LARGE_SIZE = 25
EXTRA_LARGE_SIZE = 50
# PROBLEM_SIZES = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, EXTRA_LARGE_SIZE]
PROBLEM_SIZES = [SMALL_SIZE]
SEEDS = range(0, 10)

P = [0.8, 0.75, 0.70, 0.65]
L = [0.85, 0.90, 0.95, 0.99]

MAX_ITERATION = 1500


# Forked from openai gym source
def generate_random_map(size=8, p=0.8, seed=0):
    np.random.seed(seed)
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


# size: the dimension of the grid
# p: probability that any given tile is a frozen tile
def generate_frozen_lake(size, p, seed):
    random_map = generate_random_map(size=size, p=p, seed=seed)

    env = gym.make("FrozenLake-v0", desc=random_map)
    env.seed(0)
    env.reset()
    return env

def play_episodes(enviorment, n_episodes, policy, random=False):
    """
    This fucntion plays the given number of episodes given by following a policy or sample randomly from action_space.

    Parameters:
        enviorment: openAI GYM object
        n_episodes: number of episodes to run
        policy: Policy to follow while playing an episode
        random: Flag for taking random actions. if True no policy would be followed and action will be taken randomly

    Return:
        wins: Total number of wins playing n_episodes
        total_reward: Total reward of n_episodes
        avg_reward: Average reward of n_episodes

    """
    # intialize wins and total reward
    wins = 0
    total_reward = 0

    # loop over number of episodes to play
    for episode in range(n_episodes):

        # flag to check if the game is finished
        terminated = False

        # reset the enviorment every time when playing a new episode
        state = enviorment.reset()

        while not terminated:

            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = enviorment.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward, terminated, info = enviorment.step(action)

            # enviorment.render()

            # accumalate total reward
            total_reward += reward

            # change the state
            state = next_state

            # if game is over with positive reward then add 1.0 in wins
            if terminated and reward == 1.0:
                wins += 1

    # calculate average reward
    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward

##################################################

def run_value_iteration():

    # Run VI across various problem sizes
    if False:
        convergence_times = {}
        deltas = {}
        for problem_size in PROBLEM_SIZES:
            convergence_times[problem_size] = []
            deltas[problem_size] = []
            for seed in SEEDS:
                print(f"Problem Size: {problem_size}, Seed: {seed}")
                env = generate_frozen_lake(problem_size, p=0.8, seed=seed)
                tic = time.time()
                opt_V, opt_Policy, delta = value_iteration.value_iteration(env, max_iteration=MAX_ITERATION)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[problem_size].append(elapsed_time)
                print (f"Time to converge: {elapsed_time: 0.3} ms")
                deltas[problem_size].append(delta)

        # Save the values from running
        deltas_file_pi = open('params/value_iteration/deltas', 'wb')
        pickle.dump(deltas, deltas_file_pi)
        times_file_pi = open('params/value_iteration/times', 'wb')
        pickle.dump(convergence_times, times_file_pi)

    # Run VI across various values of p
    if False:
        convergence_times = {}
        deltas = {}
        problem_size = MEDIUM_SIZE
        for p in P:
            convergence_times[p] = []
            deltas[p] = []
            for seed in SEEDS:
                print(f"p: {p}, Seed: {seed}")
                env = generate_frozen_lake(problem_size, p=p, seed=seed)
                tic = time.time()
                opt_V, opt_Policy, delta = value_iteration.value_iteration(env, max_iteration=MAX_ITERATION)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[p].append(elapsed_time)
                print(f"Time to converge: {elapsed_time: 0.3} ms")
                deltas[p].append(delta)

            # Save the values from running
            deltas_file_pi = open('params/value_iteration/p_deltas', 'wb')
            pickle.dump(deltas, deltas_file_pi)
            times_file_pi = open('params/value_iteration/p_times', 'wb')
            pickle.dump(convergence_times, times_file_pi)

    # Run value iteration across various discount factors
    if False:
        convergence_times = {}
        deltas = {}
        policies = {}
        rewards = {}
        problem_size = MEDIUM_SIZE
        p = 0.8
        L = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        for discount in L:
            convergence_times[discount] = []
            deltas[discount] = []
            policies[discount] = []
            rewards[discount] = []
            for seed in SEEDS:
                print(f"p: {p}, Seed: {seed}")
                env = generate_frozen_lake(problem_size, p=p, seed=seed)
                tic = time.time()
                opt_V, opt_Policy, delta = value_iteration.value_iteration(env, discount_factor=discount, max_iteration=MAX_ITERATION)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[discount].append(elapsed_time)
                print(f"Time to converge: {elapsed_time: 0.3} ms")
                deltas[discount].append(delta)
                policies[discount].append(opt_Policy)

                wins, total_reward, average_reward = play_episodes(env, 300, opt_Policy, False)
                print(f"Average reward: {total_reward}")
                rewards[discount].append(total_reward)

            # Save the values from running
            deltas_file_pi = open('params/value_iteration/d_deltas', 'wb')
            pickle.dump(deltas, deltas_file_pi)
            times_file_pi = open('params/value_iteration/d_times', 'wb')
            pickle.dump(convergence_times, times_file_pi)
            policies_file_pi = open('params/value_iteration/d_policies', 'wb')
            pickle.dump(policies, policies_file_pi)
            rewards_file_pi = open('params/value_iteration/d_rewards', 'wb')
            pickle.dump(rewards, rewards_file_pi)


    # Plot things
    if True:
        deltas_filehandler = open('params/value_iteration/deltas', 'rb')
        deltas = pickle.load(deltas_filehandler)

        times_filehandler = open('params/value_iteration/times', 'rb')
        convergence_times = pickle.load(times_filehandler)

        rewards_filehandler = open('params/value_iteration/d_rewards', 'rb')
        rewards = pickle.load(rewards_filehandler)


        for size in PROBLEM_SIZES:
            print(f"Average time to converge: {np.mean(convergence_times[size])} ms, std: {np.std(convergence_times[size])} ms")
        # plot_vi_convergence_size(deltas)
        # plot_vi_convergence_p(deltas)
        # plot_vi_convergence_d(deltas)
        plot_vi_convergence_dr(rewards)


def run_policy_iteration():

    # Run PI across various problem sizes
    if False:
        convergence_times = {}
        deltas = {}
        for problem_size in PROBLEM_SIZES:
            convergence_times[problem_size] = []
            deltas[problem_size] = []
            for seed in SEEDS:
                print(f"Problem Size: {problem_size}, Seed: {seed}")
                env = generate_frozen_lake(problem_size, p=0.8, seed=seed)
                tic = time.time()
                opt_V, opt_policy, delta = policy_iteration.policy_iteration(env, discount_factor=0.999, max_iteration=10000)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[problem_size].append(elapsed_time)
                print (f"Time to converge: {elapsed_time: 0.3} ms")
                deltas[problem_size].append(delta)

        # Save the values from running
        deltas_file_pi = open('params/policy_iteration/deltas', 'wb')
        pickle.dump(deltas, deltas_file_pi)
        times_file_pi = open('params/policy_iteration/times', 'wb')
        pickle.dump(convergence_times, times_file_pi)

    # Run policy iteration across various discount factors
    if True:
        convergence_times = {}
        deltas = {}
        policies = {}
        rewards = {}
        problem_size = MEDIUM_SIZE
        p = 0.8
        L = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        # L = [0.70, 0.80, 0.90, 0.99]
        for discount in L:
            convergence_times[discount] = []
            deltas[discount] = []
            policies[discount] = []
            rewards[discount] = []
            for seed in SEEDS:
                print(f"p: {p}, Seed: {seed}")
                env = generate_frozen_lake(problem_size, p=p, seed=seed)
                tic = time.time()
                opt_V, opt_policy, delta = policy_iteration.policy_iteration(env, discount_factor=discount, max_iteration=2000)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[discount].append(elapsed_time)
                print(f"Time to converge: {elapsed_time: 0.3} ms")
                deltas[discount].append(delta)
                policies[discount].append(opt_policy)

                wins, total_reward, average_reward = play_episodes(env, 300, opt_policy, False)
                print(f"Average reward: {total_reward}")
                rewards[discount].append(total_reward)

            # Save the values from running
            pickle.dump(deltas, open('params/policy_iteration/d_deltas', 'wb'))
            pickle.dump(convergence_times, open('params/policy_iteration/d_times', 'wb'))
            pickle.dump(policies, open('params/policy_iteration/d_policies', 'wb'))
            pickle.dump(rewards, open('params/policy_iteration/d_rewards', 'wb'))

    # Plot things
    if True:
        deltas = pickle.load(open('params/policy_iteration/d_deltas', 'rb'))

        convergence_times = pickle.load(open('params/policy_iteration/times', 'rb'))

        rewards = pickle.load(open('params/value_iteration/d_rewards', 'rb'))

        for size in PROBLEM_SIZES:
            print(f"Average time to converge: {np.mean(convergence_times[size])} ms, std: {np.std(convergence_times[size])} ms")
        # plotting.plot_pi_convergence_size(deltas)
        # plotting.plot_pi_convergence_d(deltas)
        plotting.plot_pi_convergence_dr(rewards)

def run_q_learning():

    # decay_rates= [0.00005, 0.0005, 0.005, 0.05, 0.5]
    # learning_rates = [0.01, 0.1, 0.5, 0.8]
    # gammas = [0.9, 0.95, 0.99]

    # Run QL HP tuning
    if False:
        convergence_times = {}
        total_rewards = {}
        for problem_size in PROBLEM_SIZES:
            convergence_times[problem_size] = []
            total_rewards[problem_size] = {}
            for seed in SEEDS:
                total_rewards[problem_size][seed] = {}
                for learning_rate in  learning_rates:
                    for gamma in gammas:
                        for decay_rate in decay_rates:
                            print(f"Problem Size: {problem_size}, Seed: {seed}, Learning Rate: {learning_rate}, Gamma {gamma}, Decay Rate: {decay_rate}")
                            env = generate_frozen_lake(problem_size, p=0.8, seed=seed)
                            tic = time.time()
                            rewards = q_learning.q_learning(env, learning_rate=learning_rate, gamma=gamma, decay_rate=decay_rate)

                            toc = time.time()
                            elapsed_time = (toc - tic) * 1000
                            convergence_times[problem_size].append(elapsed_time)
                            print(f"Time to converge: {elapsed_time: 0.3} ms")
                            # total_rewards[problem_size][seed][f"LR:{learning_rate},G:{gamma},DR:{decay_rate}"] = rewards
                            total_rewards[problem_size][seed][(learning_rate, gamma, decay_rate)] = rewards

        # Save the values from running
        pickle.dump(total_rewards, open('params/q_learning/total_rewards', 'wb'))

    gamma = 0.9999
    learning_rate = 0.8
    decay_rate = 0.0005
    PROBLEM_SIZES = [MEDIUM_SIZE]
    MAX_EPISODES = 25000

    # Run QL
    if False:
        convergence_times = {}
        total_rewards = {}
        for problem_size in PROBLEM_SIZES:
            convergence_times[problem_size] = []
            total_rewards[problem_size] = {}
            for seed in SEEDS:
                print(f"Problem Size: {problem_size}, Seed: {seed}, Learning Rate: {learning_rate}, Gamma {gamma}, Decay Rate: {decay_rate}")
                env = generate_frozen_lake(problem_size, p=0.99, seed=seed)
                tic = time.time()
                rewards, epsilons = q_learning.q_learning(env, learning_rate=learning_rate, gamma=gamma, decay_rate=decay_rate, total_episodes=MAX_EPISODES)

                toc = time.time()
                elapsed_time = (toc - tic) * 1000
                convergence_times[problem_size].append(elapsed_time)
                print(f"Time to converge: {elapsed_time: 0.3} ms")
                total_rewards[problem_size][seed] = rewards

        # Save the values from running
        pickle.dump(total_rewards, open('params/q_learning/s_big_easy_total_rewards', 'wb'))
        pickle.dump(epsilons, open('params/q_learning/s_big_easy_epsilons', 'wb'))


    # Plot things
    if True:
        total_rewards = pickle.load(open('params/q_learning/s_big_easy_total_rewards', 'rb'))
        epsilons = pickle.load(open('params/q_learning/s_big_easy_epsilons', 'rb'))

        plotting.plot_episode_rewards(total_rewards, epsilons, 1000)


######
# run_value_iteration()
# run_policy_iteration()
run_q_learning()