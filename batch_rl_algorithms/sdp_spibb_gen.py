import numpy as np


class SDPSPIBBGen:
    NAME = 'SDPSPIBBGen'

    def __init__(self, env, pi_b, nb_states, nb_actions, MLE_T, mask, known_states_actions,
                 initial_policy, initial_qvalues, gamma, max_nb_it=5000):
        self.env = env
        self.pi_b = pi_b
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.max_nb_it = max_nb_it
        self.MLE_T = MLE_T
        self.mask = mask
        self.known_states_actions = known_states_actions
        self.gamma = gamma
        self.initial_policy = initial_policy
        self.initial_qvalues = initial_qvalues
        self.policy = None
        self.q_values = None

    def fit(self):
        """
        Starts the actual training by reiterating between self._policy_evaluation() and self._policy_improvement()
        until convergence of the action-value function or the maximal number of iterations (self.max_nb_it) is reached.
        :return:
        """
        self.policy_iteration()

    def policy_iteration(self, epsilon=0.000000001, max_iterations=10000):

        policy = self.initial_policy.copy()
        q_values = self.initial_qvalues.copy()

        max_dif = 1
        i = 0
        while max_dif > 0.001 and i < max_iterations:
            # print('max dif: %s' % max_dif)
            # print('PI it: %s' % i)
            q_values = self.policy_evaluation(q_values, policy, epsilon, 100)
            new_policy = self.policy_improvement(q_values, policy)  # SPI
            max_dif = 0
            for s in self.known_states_actions.keys():
                max_dif = max(abs(policy[s] - new_policy[s]).max(), max_dif)

            policy = new_policy.copy()
            i += 1

        self.policy = policy.copy()
        self.q_values = q_values.copy()

    def policy_evaluation(self, initial_q, initial_policy, epsilon=0.000000001, max_iterations=100):

        q_values = initial_q.copy()

        q_prime = initial_q.copy()

        i = 0
        while i < max_iterations:
            # print('PE it: %s' % i)
            policy = self.policy_improvement(q_values, initial_policy)  # SPI
            for state in self.known_states_actions.keys():
                for action in self.known_states_actions[state]:
                    delayed_reward = 0
                    if (state not in self.MLE_T.keys()) or (action not in self.MLE_T[state].keys()):
                        ns = [state]
                    else:
                        ns = list(self.MLE_T[state][action].keys())
                    for next_state in ns:
                        if next_state in q_values.keys():
                            delayed_reward += np.sum(self.MLE_T[state][action][next_state] *
                                                     q_values[next_state] *
                                                     policy[next_state])
                        else:
                            delayed_reward = 0
                    # q_prime[state, action] = np.sum(self.env.reward_function(state, action)) + self.gamma * delayed_reward
                    q_prime[state][int(action)] = np.sum(self.env.reward_function(state, action)) + self.gamma * delayed_reward
            q_values = q_prime.copy()
            i += 1

        return q_values

    def policy_improvement(self, q_values, policy):
        """
        Updates the current policy self.pi.
        """

        pi = policy.copy()

        for s in self.known_states_actions.keys():
            p_non_boot = np.sum(pi[s][self.mask[s]])
            if p_non_boot > 0:
                index_non_boot = np.where(self.mask[s])[0]
                pi[s][index_non_boot] = 0
                max_index = np.argmax(q_values[s][index_non_boot])
                pi[s][index_non_boot[max_index]] = p_non_boot

        return pi

    def get_distribution_generative(self):
        pi_new = self.policy.copy()
        for s in pi_new.keys():
            pi_new[s] = np.zeros(self.nb_actions)
            pi_new[s][self.policy[s].argmax(axis=0)] = 1

        return pi_new
