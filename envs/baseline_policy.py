import random

import numpy as np


class SysAdminGenerativeBaselinePolicy:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.nb_states = env.n_states
        self.nb_actions = env.n_actions
        self.p = 0.7

    def generative_baseline(self, state, prob):
        state = np.array(list(self.env.decode(state)))
        machines_on = np.where(state == 1)[0]
        machines_off = np.where(state == 0)[0]
        pi = np.zeros(self.env.n_actions)

        if machines_on.size == self.env.n_machines:
            pi[self.env.n_actions - 1] = self.p
            pi[:self.env.n_actions - 1] = (1 - self.p) / (self.env.n_actions - 1)

        elif machines_off.size == self.env.n_machines:
            pi[0] = self.p
            indices = [i for i, x in enumerate(pi) if x == 0]
            probs = [(1 - self.p) / (len(pi) - 1)] * len(indices)
            pi[indices] = probs

        else:
            for s in machines_on:
                if (s + 1) % (self.env.n_actions - 1) in machines_off:
                    pi[s + 1] = self.p
                    indices = [i for i, x in enumerate(pi) if i != s + 1]
                    probs = [(1 - self.p) / len(indices)] * len(indices)
                    pi[indices] = probs
                    break
                elif (s - 1) % (self.env.n_actions - 1) in machines_off:
                    pi[s - 1] = self.p
                    indices = [i for i, x in enumerate(pi) if i != s - 1]
                    probs = [(1 - self.p) / len(indices)] * len(indices)
                    pi[indices] = probs
                    break
                continue
        pi /= pi.sum()

        if prob:
            return pi
        else:
            return np.random.choice(pi.shape[0], p=pi)


class TaxiAbsorbingBaselinePolicy:
    def __init__(self, env, gamma, initial_state, max_nb_it=1000):
        self.env = env
        self.gamma = gamma
        self.initial_state = initial_state
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.q = np.zeros((self.nb_states, self.nb_actions))
        self.max_nb_it = max_nb_it
        self.accuracy = 0.7
        self.compute_baseline()

    def compute_baseline(self):
        old_q = np.ones([self.nb_states, self.nb_actions])
        self.nb_it = 0

        while np.linalg.norm(self.q - old_q) > 10 ** (-9) and self.nb_it < self.max_nb_it:
            self.nb_it += 1
            old_q = self.q.copy()
            self._policy_evaluation()
            self._policy_improvement()
        pi_rand = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.pi = self.accuracy * self.pi + (1 - self.accuracy) * pi_rand

    def _policy_improvement(self):
        """
        Updates the current policy self.pi (Here: greedy update).
        """
        self.pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            self.pi[s, np.argmax(self.q[s, :])] = 1

    def _policy_evaluation(self):
        """
        Computes the action-value function for the current policy self.pi.
        """
        nb_sa = self.nb_actions * self.nb_states
        M = np.eye(nb_sa) - self.gamma * np.einsum('ijk,kl->ijkl', self.env.T, self.pi).reshape(nb_sa, nb_sa)
        self.q = np.linalg.solve(M, self.env.R.reshape(nb_sa)).reshape(self.nb_states, self.nb_actions)


class ScalableTaxiAbsorbingBaselinePolicy:
    def __init__(self, env, gamma, initial_state, max_nb_it=1000):
        self.env = env
        self.gamma = gamma
        self.initial_state = initial_state
        _, _, self.pass_loc, self.dest_idx = self.env.decode(self.initial_state)
        self.target_idx = self.pass_loc
        self.target_loc = self.env.locs[self.target_idx]
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.pi = np.ones((self.nb_states, self.nb_actions)) / self.nb_actions
        self.q = np.zeros((self.nb_states, self.nb_actions))
        self.max_nb_it = max_nb_it
        self.accuracy = 0.9
        # self.heuristic = ['row', 'col']
        self.heuristic = ['row']

        self.go_to_the_target = False

    def prob_pi_b(self, state):
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.decode(state)
        taxi_loc = (taxi_row, taxi_col)
        # heuristic = self.heuristic[random.randint(0, 1)]
        heuristic = self.heuristic[random.randint(0, 0)]

        if heuristic == 'row':  # Change row first
            if taxi_row == self.target_loc[0]:
                # Target cell
                if taxi_col == self.target_loc[1]:
                    # print(self.go_to_the_target)
                    if self.go_to_the_target and taxi_loc == self.env.locs[self.dest_idx]:
                        # print(taxi_col)
                        # print((self.target_loc[0], self.target_loc[1]))
                        # print(self.env.locs[self.target_idx])
                        action = np.zeros(self.nb_actions)
                        action[5] = 1.0
                        self.go_to_the_target = False
                        self.target_idx = self.pass_loc
                        self.target_loc = self.env.locs[self.target_idx]
                    else:
                        action = np.zeros(self.nb_actions)
                        action[4] = 1.0
                        self.go_to_the_target = True
                        self.target_idx = self.dest_idx
                        self.target_loc = self.env.locs[self.target_idx]
                # The target is on the right
                elif taxi_col < self.target_loc[1]:
                    action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                    action[4] = 0
                    action[5] = 0
                    action[2] = self.accuracy
                # The target is on the left
                else:
                    action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                    action[4] = 0
                    action[5] = 0
                    action[3] = self.accuracy
            elif taxi_row < self.target_loc[0]:
                action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                action[4] = 0
                action[5] = 0
                action[0] = self.accuracy
            else:
                action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                action[4] = 0
                action[5] = 0
                action[1] = self.accuracy
        else:  # Change coloumn first
            if taxi_col == self.target_loc[1]:
                # Target cell
                if taxi_row == self.target_loc[0]:
                    if self.go_to_the_target:
                        action = np.zeros(self.nb_actions)
                        action[5] = 1.0
                        self.go_to_the_target = False
                        self.target_idx = self.pass_loc
                        self.target_loc = self.env.locs[self.target_idx]
                    else:
                        action = np.zeros(self.nb_actions)
                        action[4] = 1.0
                        self.go_to_the_target = True
                        self.target_idx = self.dest_idx
                        self.target_loc = self.env.locs[self.target_idx]
                # The target is down
                elif taxi_row < self.target_loc[0]:
                    action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                    action[4] = 0
                    action[5] = 0
                    action[0] = self.accuracy
                # The target is up
                else:
                    action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                    action[4] = 0
                    action[5] = 0
                    action[1] = self.accuracy
            elif taxi_col < self.target_loc[1]:
                action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                action[4] = 0
                action[5] = 0
                action[2] = self.accuracy
            else:
                action = np.full(self.nb_actions, (1 - self.accuracy) / (self.nb_actions - 3))
                action[4] = 0
                action[5] = 0
                action[3] = self.accuracy

        return np.array(action)