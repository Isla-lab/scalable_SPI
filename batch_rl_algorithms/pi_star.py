from batch_rl_algorithms.batch_rl_algorithm import BatchRLAlgorithm

import numpy as np


class PiStar(BatchRLAlgorithm):
    # This class implements Dynamic Programming as described by 'Reinforcement Learning - An Introduction' by Sutton and
    # Barto by using the true reward matrix and transition probabilities. Even though it is not a Batch RL algorithm
    # this class inherits from BatchRLAlgorithm to be able to reuse its PE and PI step and also otherwise fit into
    # the framework, to make it easier to include the optimal policies in experiments.
    NAME = 'PI_STAR'

    def __init__(self, pi_b, gamma, nb_states, nb_actions, data, R, P, episodic, checks=False, zero_unseen=True,
                 max_nb_it=5000, speed_up_dict=None):
        """
        As this class does not really implement a Batch RL algorithm, some of the input parameters can be set to None
        :param pi_b: not necessary, choice is not important
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: not necessary, choice is not important
        :param R: reward matrix as numpy array with shape (nb_states, nb_states), assuming that the reward is
        deterministic w.r.t. the
         previous and the next states
        :param P: true transition probabilities as numpy array with shape (nb_states, nb_actions, nb_states)
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: not necessary, choice is not important
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: not necessary, choice is not important
        """
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.zero_unseen = zero_unseen
        self.episodic = episodic
        self.max_nb_it = max_nb_it
        self.pi = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions
        self.q = np.zeros([nb_states, nb_actions])
        self.R_state_state = R
        self.checks = checks
        self.transition_model = P
        self._compute_R_state_action()


class generative_PiStar():
    # This class implements Dynamic Programming as described by 'Reinforcement Learning - An Introduction' by Sutton and
    # Barto by using the true reward matrix and transition probabilities. Even though it is not a Batch RL algorithm
    # this class inherits from BatchRLAlgorithm to be able to reuse its PE and PI step and also otherwise fit into
    # the framework, to make it easier to include the optimal policies in experiments.
    NAME = 'generative_PI_STAR'

    def __init__(self, env, gamma, nb_states, nb_actions, max_nb_it=5000):
        """
        As this class does not really implement a Batch RL algorithm, some of the input parameters can be set to None
        :param pi_b: not necessary, choice is not important
        :param gamma: discount factor
        :param nb_states: number of states of the MDP
        :param nb_actions: number of actions available in each state
        :param data: not necessary, choice is not important
        :param R: reward matrix as numpy array with shape (nb_states, nb_states), assuming that the reward is
        deterministic w.r.t. the
         previous and the next states
        :param P: true transition probabilities as numpy array with shape (nb_states, nb_actions, nb_states)
        :param episodic: boolean variable, indicating whether the MDP is episodic (True) or non-episodic (False)
        :param zero_unseen: not necessary, choice is not important
        :param max_nb_it: integer, indicating the maximal number of times the PE and PI step should be executed, if
        convergence is not reached
        :param checks: boolean variable indicating if different validity checks should be executed (True) or not
        (False); this should be set to True for development reasons, but it is time consuming for big experiments
        :param speed_up_dict: not necessary, choice is not important
        """
        self.env = env
        self.gamma = gamma
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.max_nb_it = max_nb_it
        self.pi = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions
        self.q = np.zeros([nb_states, nb_actions])

    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        self.pi = self.policy_iteration()

    def policy_improvement(self, q_values):
        """
        Updates the current policy self.pi.
        """
        # print('Performing PI')
        pi = np.zeros((self.nb_states, self.nb_actions))
        greedy_action_indices = q_values.argmax(axis=1)
        pi[np.arange(self.nb_states), greedy_action_indices] = 1

        return pi

    def value_iteration(self, epsilon=0.000000001, max_iterations=100, initial_q=None):

        q_values = np.zeros(shape=(self.nb_states, self.nb_actions))
        q_prime = np.zeros(shape=(self.nb_states, self.nb_actions), dtype=float)

        res = epsilon + 1
        i = 0
        while res > epsilon and i < max_iterations:
            print('VI it: %s, res: %s' % (i, res))
            policy = self.policy_improvement(q_values)  # PI
            for s in range(self.nb_states):
                # print('State %s' %s)
                for a in range(self.nb_actions):
                    next_state_dist = self.get_next_states_dist(s, a)
                    if np.count_nonzero(next_state_dist) > 1:
                        exp_future_reward = np.sum(next_state_dist[np.newaxis].T * q_values * policy)
                    else:
                        exp_future_reward = 0
                        p, ns = self.get_successors_of(s, a)
                        for next_state in range(len(ns)):
                            exp_future_reward += np.sum(q_values[ns[next_state]] * policy[ns[next_state]]
                                                        * p[next_state])
                    q_prime[s, a] = self.env.get_reward_function(s, a) + self.gamma * exp_future_reward
            res = np.max(np.abs(q_values - q_prime))

            q_values = np.copy(q_prime)
            i += 1

        return q_values

    def policy_iteration(self, epsilon=0.000000001, max_iterations=100000000, initial_policy=None, initial_q=None):
        if initial_q is None:
            q_values = np.zeros((self.nb_states, self.nb_actions))
        else:
            q_values = np.copy(initial_q)  # copy q-values of behavior policy

        if initial_policy is None:
            policy = self.policy_improvement(q_values)
        else:
            policy = initial_policy

        max_dif = 1
        i = 0

        while max_dif > 0.001 and i < max_iterations:
            # print('PI it: %s' % i)

            q_values = self.value_iteration(epsilon, max_iterations, initial_q=q_values)

            new_policy = self.policy_improvement(q_values)  # PI
            max_dif = 0
            for s in range(self.nb_states):
                max_dif = max(abs(policy[s] - new_policy[s]).max(), max_dif)

            policy = new_policy

            i += 1

        return policy

    def get_next_states_dist(self, s, a):
        return self.env.get_next_states_dist(s, a)

    def get_successors_of(self, s, a):
        return self.env.get_successors(s, a)