import numpy as np

def value_iteration(env, discount_factor=0.999, max_iteration=1000):
    """
    Algorithm to solve MPD.

    Arguments:
        env: openAI GYM Enviorment object.
        discount_factor: MDP discount factor.
        max_iteration: Maximum No.  of iterations to run.

    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        optimal_policy: Optimal policy. Vector of length nS.

    """
    # intialize value fucntion
    V = np.zeros(env.nS)
    delta = []

    # iterate over max_iterations
    for i in range(max_iteration):

        #  keep track of change with previous value function
        prev_v = np.copy(V)

        # loop over all states
        for state in range(env.nS):
            # Asynchronously update the state-action value
            # action_values = one_step_lookahead(env, state, V, discount_factor)

            # Synchronously update the state-action value
            action_values = one_step_lookahead(env, state, prev_v, discount_factor)

            # select best action to perform based on highest state-action value
            best_action_value = np.max(action_values)

            # update the current state-value fucntion
            V[state] = best_action_value

        delta.append(np.sum(np.abs(V-prev_v)))

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if (np.all(np.isclose(V, prev_v))):
                print('Value converged at iteration %d' % (i + 1))
                break

    # intialize optimal policy
    optimal_policy = np.zeros(env.nS, dtype='int8')

    # update the optimal polciy according to optimal value function 'V'
    optimal_policy = update_policy(env, optimal_policy, V, discount_factor)

    return V, optimal_policy, delta


def one_step_lookahead(env, state, V, discount_factor=0.99):
    """
    Helper function to  calculate state-value function

    Arguments:
        env: openAI GYM Enviorment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """

    # initialize vector of action values
    action_values = np.zeros(env.nA)

    # loop over the actions we can take in an enviorment
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, info in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probablity * (reward + (discount_factor * V[next_state]))

    return action_values

def update_policy(env, policy, V, discount_factor):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        env: openAI GYM Enviorment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """

    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)

        # choose the action which maximizes the state-action value.
        policy[state] = np.argmax(action_values)

    return policy

