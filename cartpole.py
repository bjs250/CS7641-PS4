# Finite-state MDP solved using Value Iteration

import numpy as np
import gym
import pickle

import plotting

env = gym.make('CartPole-v0')
observation = env.reset()

num_observation_dimensions = np.size(observation)
num_actions = env.action_space.n

observation_space_high = np.array([4.8, 2.4, 0.418, 2.4])
observation_space_low = np.array([-4.8, -2.4, -0.418, -2.4])

def make_observation_bins(min, max, num_bins):
    if (min < -1000):
        min = -5  # Should really learn this const instead
    if (max > 1000):
        max = 5

    bins = np.arange(min, max, (float(max) - float(min)) / ((num_bins) - 2))
    bins = np.sort(np.append(bins, [0]))  # Ensure we split at 0

    return bins



def observation_to_state(observation):
    state = 0
    for observation_dimension in range(num_observation_dimensions):
        state = state + np.digitize(observation[observation_dimension],
                                    observation_dimension_bins[observation_dimension]) \
                * pow(num_bins_per_observation_dimension, observation_dimension)

    return state


# print("Sense Check: Min State: {} Max State: {} Num States: {}".format(observation_to_state([-5, -5, -5, -5.5]),
#                                                                  observation_to_state([5, 5, 5, 5.5]),
#                                                                  num_states))
#
# state_values = np.random.rand(num_states) * 0.1
# state_rewards = np.zeros((num_states))
# state_transition_probabilities = np.ones((num_states, num_states, num_actions)) / num_states
# state_transition_counters = np.zeros((num_states, num_states, num_actions))


def pick_best_action(current_state, state_values, state_transition_probabilities):
    best_action = -1
    best_action_value = -np.Inf
    for a_i in range(num_actions):
        action_value = state_transition_probabilities[current_state, :, a_i].dot(state_values)
        if (action_value > best_action_value):
            best_action_value = action_value
            best_action = a_i
        elif (action_value == best_action_value):
            if np.random.randint(0, 2) == 0:
                best_action = a_i

    return best_action


def update_state_transition_probabilities_from_counters(probabilities, counters):
    for a_i in range(num_actions):
        for s_i in range(num_states):
            total_transitions_out_of_state = np.sum(counters[s_i, :, a_i])
            if (total_transitions_out_of_state > 0):
                probabilities[s_i, :, a_i] = counters[s_i, :, a_i] / total_transitions_out_of_state

    return probabilities


def run_value_iteration(state_values, state_transition_probabilities, state_rewards):
    print("running VI")
    gamma = 0.995
    convergence_tolerance = 0.02
    iteration = 0
    MAX_DIF_VALUE = 1000
    max_dif = MAX_DIF_VALUE
    deltas = []
    while max_dif > convergence_tolerance:
        # print(f"iteration: {iteration}, {max_dif}, {convergence_tolerance}")
        iteration = iteration + 1
        old_state_values = np.copy(state_values)

        best_action_values = np.zeros(num_states) - MAX_DIF_VALUE
        for a_i in range(num_actions):
            best_action_values = \
                np.maximum(best_action_values, state_transition_probabilities[:, :, a_i].dot(state_values))

        state_values = state_rewards + gamma * best_action_values
        max_dif = np.max(np.abs(state_values - old_state_values))
        deltas.append(max_dif)

    print("VI done")
    return state_values, deltas

######

if True:

    if True:
        SIZES = [4, 6, 8, 10]
        deltas = {}

        for size in SIZES:
            # Hyperparameter
            num_bins_per_observation_dimension = size  # Could try different number of bins for the different dimensions

            observation_dimension_bins = []
            for observation_dimension in range(num_observation_dimensions):
                observation_dimension_bins.append(make_observation_bins(observation_space_low[observation_dimension], \
                                                                        observation_space_high[observation_dimension], \
                                                                        num_bins_per_observation_dimension))

            num_states = pow(num_bins_per_observation_dimension, num_observation_dimensions)
            print(f"num states: {num_states}")

            state_values = np.random.rand(num_states)
            state_rewards = np.zeros(num_states)
            state_transition_probabilities = np.ones((num_states, num_states, num_actions)) / num_states
            state_transition_counters = np.zeros((num_states, num_states, num_actions))

            episode_rewards = []
            deltas[size] = {}
            for i_episode in range(10):
                current_observation = env.reset()
                current_state = observation_to_state(current_observation)

                episode_reward = 0

                for t in range(1000):
                    print(f"Size: {size}, Seed: {i_episode}, Step:{t}")
                    action = pick_best_action(current_state, state_values, state_transition_probabilities)
                    old_state = current_state
                    observation, reward, done, info = env.step(action)
                    current_state = observation_to_state(observation)

                    state_transition_counters[old_state, current_state, action] = \
                        state_transition_counters[old_state, current_state, action] + 1

                    episode_reward = episode_reward + reward

                    if done:
                        episode_rewards.append(episode_reward)
                        print("Reward: {}, Average reward over {} trials: {}".format(episode_reward, i_episode,
                                                                               np.mean(episode_rewards[-100:])))

                        if (t < 195):
                            reward = -1
                        else:
                            reward = 0
                        state_rewards[current_state] = reward

                        state_transition_probabilities = update_state_transition_probabilities_from_counters(
                            state_transition_probabilities, state_transition_counters)
                        state_values, episode_deltas = run_value_iteration(state_values, state_transition_probabilities, state_rewards)
                        deltas[size][i_episode] = episode_deltas
                        break

        pickle.dump(deltas, open(f"params/value_iteration/cartpole_4567", 'wb'))

    if True:
        deltas = pickle.load(open(f"params/value_iteration/cartpole_4567", 'rb'))
        plotting.plot_cartpole_vi_deltas(deltas)

if False:
    episode_rewards = []
    for i_episode in range(2000):
        current_observation = env.reset()
        current_state = observation_to_state(current_observation)

        episode_reward = 0

        for t in range(1000):
            print(f"Episode: {i_episode}, Step:{t}")
            action = pick_best_action(current_state, state_values, state_transition_probabilities)
            old_state = current_state
            observation, reward, done, info = env.step(action)
            current_state = observation_to_state(observation)

            state_transition_counters[old_state, current_state, action] = \
                state_transition_counters[old_state, current_state, action] + 1

            episode_reward = episode_reward + reward

            if done:
                episode_rewards.append(episode_reward)
                print("Reward: {}, Average reward over {} trials: {}".format(episode_reward, i_episode,
                                                                       np.mean(episode_rewards[-100:])))

                if (t < 195):
                    reward = -1
                else:
                    reward = 0
                state_rewards[current_state] = reward

                state_transition_probabilities = update_state_transition_probabilities_from_counters(
                    state_transition_probabilities, state_transition_counters)
                state_values = run_value_iteration(state_values, state_transition_probabilities, state_rewards)
                break
    pickle.dump(episode_rewards, open(f"params/value_iteration/cartpole_{num_bins_per_observation_dimension}", 'wb'))

# episode_rewards = pickle.load(open(f"params/value_iteration/cartpole_{num_bins_per_observation_dimension}", 'rb'))
# plotting.plot_cartpole_vi_reward(episode_rewards, 100)