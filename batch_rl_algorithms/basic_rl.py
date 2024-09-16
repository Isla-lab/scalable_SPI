import numpy as np


class Basic_rl:
    # Implements Dynamic Programming as explained in 'Reinforcement Learning - An Introduction' by Sutton and Barto
    # using the estimated MDP.
    NAME = 'Basic_rl'

    def __init__(self, env, nb_states, nb_actions, MLE_T, initial_policy, initial_qvalues, gamma, max_nb_it=5000):

        self.env = env
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.max_nb_it = max_nb_it
        self.MLE_T = MLE_T
        self.gamma = gamma
        self.initial_policy = 1 / self.nb_actions * np.ones([self.nb_states, self.nb_actions])
        self.initial_qvalues = np.zeros([self.nb_states, self.nb_actions])
        self.pi_improved = None
        self.q_values = None

    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        self.policy_iteration()

    def policy_iteration(self, epsilon=0.000000001, max_iterations=100000000):

        policy = self.initial_policy.copy()
        q_values = self.initial_qvalues.copy()

        max_dif = 1
        i = 0
        while max_dif > 0.001 and i < max_iterations:
            print('max dif: %s' % max_dif)
            print('PI it: %s' % i)
            q_values = self.policy_evaluation(q_values, policy, epsilon, 100)
            new_policy = self.policy_improvement(q_values, policy)  # SPI
            max_dif = 0
            for s in range(self.nb_states):
                max_dif = max(abs(policy[s] - new_policy[s]).max(), max_dif)

            policy = new_policy.copy()
            i += 1

        self.pi_improved = policy.copy()
        self.q_values = q_values.copy()

    def policy_evaluation(self, initial_q, initial_policy, epsilon=0.000000001, max_iterations=100000000):

        q_values = initial_q.copy()

        q_prime = initial_q.copy()

        i = 0
        while i < max_iterations:
            print('PE it: %s' % i)
            policy = self.policy_improvement(q_values, initial_policy)  # SPI
            for state in range(self.nb_states):
                for action in range(self.nb_actions):
                    delayed_reward = 0
                    for next_state in range(self.nb_states):
                        if self.MLE_T[state][action][next_state] != 0.0:
                            delayed_reward += np.sum(self.MLE_T[state][action][next_state] *
                                                     q_values[next_state] *
                                                     policy[next_state])
                        else:
                            delayed_reward = 0
                    q_prime[state, action] = np.sum(
                        self.env.reward_function(state, action)) + self.gamma * delayed_reward
            q_values = q_prime.copy()
            i += 1

        return q_values

    def policy_improvement(self, q_values, policy):
        """
        Updates the current policy self.pi.
        """

        pi = policy.copy()

        for s in range(self.nb_states):
            max_index = np.argmax(q_values[s])
            pi[s][max_index] = 1.0

        return pi

    def get_distribution(self):
        pi_new = self.pi_improved.copy()
        for s in range(self.nb_states):
            pi_new[s] = np.zeros(self.nb_actions)
            pi_new[s][self.pi_improved[s].argmax(axis=0)] = 1

        return pi_new