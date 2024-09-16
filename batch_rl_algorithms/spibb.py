# authors: anonymized

import numpy as np
from scipy.optimize import linprog


# Computes the non-bootstrapping mask
def compute_mask(nb_states, nb_actions, epsilon, delta, batch):
    N_wedge = 2*(np.log((2*nb_states*nb_actions*2**nb_states)/delta))/epsilon**2
    return compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch), N_wedge


def compute_mask_N_wedge(nb_states, nb_actions, N_wedge, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
    # for [action, state, next_state] in batch:
        count_state_action[state, action] += 1
    return count_state_action > N_wedge, count_state_action


# Computes the transition errors for all state-action pairs
def compute_errors(nb_states, nb_actions, delta, batch):
    count_state_action = np.zeros((nb_states, nb_actions))
    errors = np.zeros((nb_states, nb_actions))
    for [action, state, next_state, reward] in batch:
        count_state_action[state, action] += 1
    for state in range(nb_states):
        for action in range(nb_actions):
            if count_state_action[state, action] == 0:
                errors[state, action] = np.inf
            else:
                errors[state, action] = np.sqrt(
                    2*(np.log(2*(nb_states*nb_actions)/delta))/count_state_action[state, action]
                )
    return errors


def policy_evaluation_exact(pi, r, p, gamma):
    """
    Evaluate policy by taking the inverse
    Args:
    pi: policy, array of shape |S| x |A|
    r: the true rewards, array of shape |S| x |A|
    p: the true state transition probabilities, array of shape |S| x |A| x |S|
    Return:
    v: 1D array with updated state values
    """
    # Rewards according to policy: Hadamard product and row-wise sum
    r_pi = np.einsum('ij,ij->i', pi, r)

    # Policy-weighted transitions:
    # multiply p by pi by broadcasting pi, then sum second axis
    # result is an array of shape |S| x |S|
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
    return v, r + gamma*np.einsum('i, jki->jk', v, p)


def state_action_density(pi, p):
    x_0 = np.zeros(pi.shape[0])
    x_0[0] = 1
    p_pi = np.einsum('ijk, ij->ik', p, pi)
    d = np.dot(x_0, np.linalg.inv((np.eye(p_pi.shape[0]) - p_pi)))
    print(d)
    print(d.sum())
    dxa = np.minimum(1,np.einsum('i, ij->ij', d, pi))
    dxaf = dxa[:,2:]
    xs = [1,2,3,5,7,10,15,20,30,50,70,100,150,200,300,500,700,1000,1500,2000,3000,5000,7000,10000]
    ys = []
    zs = []
    for x in xs:
        y = (x*dxa*(1-dxa)**(x-1)).sum()
        z = (x*dxaf*(1-dxaf)**(x-1)).sum()
        ys.append(y)
        zs.append(z)
    print(ys)
    print(zs)
    print(dxa)
    print(dxaf)


def softmax(q, temp):
    exp = np.exp(temp*(q - np.max(q, axis=1)[:,None]))
    pi = exp / np.sum(exp, axis=1)[:,None]
    return pi


class spibb():
    # gamma is the discount factor,
    # nb_states is the number of states in the MDP,
    # nb_actions is the number of actions in the MDP,
    # pi_b is the baseline policy,
    # mask is the mask where the one does not need to bootstrap,
    # model is the transition model,
    # reward is the reward model,
    # space denotes the type of policy bootstrapping,
    # q_pib_est is the MC estimator of the state values of baseline policy,
    # errors contains the errors on the probabilities or Q-function,
    # max_nb_it is the maximal number of policy improvement
    NAME = 'spibb'

    def __init__(self, gamma, nb_states, nb_actions, pi_b, mask, model, reward, space,
                q_pib_est=None, errors=None, epsilon=None, max_nb_it=1e5, version=1):
        self.version = version
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.P = model
        self.pi_b = pi_b
        # if version == 1:
        #     self.pi_b_masked = self.pi_b.copy()
        #     self.pi_b_masked[mask] = 0
        self.pi_b_masked = self.pi_b.copy()
        self.pi_b_masked[mask] = 0
        self.mask = mask
        if version == 1:
            self.R = reward.reshape(self.nb_states * self.nb_actions)
        else:
            self.R = reward
        self.space = space
        self.q_pib_est_masked = None
        if q_pib_est is not None:
            self.q_pib_est_masked = q_pib_est.copy()
            self.q_pib_est_masked[mask] = 0
        self.errors = errors
        self.epsilon = epsilon
        self.max_nb_it = max_nb_it

    # starts a new episode (during the policy exploitation)
    def new_episode(self):
        self.has_bootstrapped = False

    # trains the policy
    def fit(self):
        if self.version == 1:
            pi = self.pi_b.copy()
        elif self.version == 2:
            pi = np.zeros((self.nb_states, self.nb_actions))
        q = np.zeros((self.nb_states, self.nb_actions))
        old_q = np.ones((self.nb_states, self.nb_actions))
        nb_sa = self.nb_states * self.nb_actions
        nb_it = 0
        old_pi = None
        while np.linalg.norm(q - old_q) > 0.000000001 and nb_it < self.max_nb_it:
            old_q = q.copy()
            ################## Matrix form ##################################
            if self.version == 1:
                new_matrix = np.einsum('ijk,kl->ijkl', self.P, pi).reshape(nb_sa, nb_sa)
                M = np.eye(nb_sa) - self.gamma * new_matrix
                q = np.dot(np.linalg.inv(M), self.R).reshape(self.nb_states, self.nb_actions)
            ################## Iterative non matrix form ####################
            elif self.version == 2:
                for s in range(0, self.nb_states):
                    for a in range(0, self.nb_actions):
                        qval = self.R[s,a] # R(s,a)
                        next_val = 0
                        for _s in range(0, self.nb_states):
                            qs = 0
                            for _a in range(0, self.nb_actions):
                                qs += pi[_s][_a]*q[_s][_a]
                            next_val += self.P[s, a, _s] * qs
                        q[s][a] += qval + self.gamma * next_val #or q[s][a] = qval
            else:
                raise
            #################################################################
            if self.q_pib_est_masked is not None:
                q += self.q_pib_est_masked
            pi = self.update_pi(q, old_pi)
            old_pi = pi
            nb_it += 1
            #if nb_it > 1000:
            #    with open("notconverging.txt", "a") as myfile:
            #        myfile.write(str(self.space) + " epsilon=" + str(self.epsilon) + " nb_traj=" + str(self.nb_traj) + " is not converging. \n")
            #    break
        self.pi = pi
        self.q = q
        
    # does the policy improvement inside the policy iteration loop
    def update_pi(self, q, old_pi=None):

        if self.version == 1:
            pi = self.pi_b_masked.copy()
            for s in range(self.nb_states):
                if len(q[s, self.mask[s]]) > 0:
                    pi_b_masked_sum = np.sum(self.pi_b_masked[s])
                    pi[s][np.where(self.mask[s])[0][np.argmax(q[s, self.mask[s]])]] = 1 - pi_b_masked_sum
        elif self.version == 2:
            pi = np.zeros((self.nb_states, self.nb_actions))
            for s in range(self.nb_states):
                pi_b_masked_sum = 0
                for a in range(self.nb_actions):
                    if self.mask[s][a] == False:
                        pi[s][a] = self.pi_b[s,a]
                        pi_b_masked_sum += self.pi_b[s,a]
                if len(q[s, self.mask[s]]) > 0:
                    pi[s][np.where(self.mask[s])[0][np.argmax(q[s, self.mask[s]])]] = 1 - pi_b_masked_sum

        return pi

    # implements the trained policy
    def predict(self, state, bootstrap):
        if self.has_bootstrapped:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi_b[state])
        else:
            choice = np.random.choice(self.nb_actions, 1, p=self.pi[state])
            if bootstrap and np.sum(self.P[state, choice]) < 0.5:
                self.has_bootstrapped = True
        return choice

    def policy_evaluation_exact(self, pi, r, p, gamma):
        """
        Evaluate policy by taking the inverse
        Args:
          pi: policy, array of shape |S| x |A|
          r: the true rewards, array of shape |S| x |A|
          p: the true state transition probabilities, array of shape |S| x |A| x |S|
        Return:
          v: 1D array with updated state values
        """
        # Rewards according to policy: Hadamard product and row-wise sum
        r_pi = np.einsum('ij,ij->i', pi, r)

        # Policy-weighted transitions:
        # multiply p by pi by broadcasting pi, then sum second axis
        # result is an array of shape |S| x |S|
        p_pi = np.einsum('ijk, ij->ik', p, pi)
        v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
        return v, r + gamma*np.einsum('i, jki->jk', v, p)

    def get_distribution(self):
        pi_new = self.pi.copy()
        for s in range(self.nb_states):
            pi_new[s] = np.zeros(self.nb_actions)
            pi_new[s][self.pi[s].argmax(axis=0)] = 1

        return pi_new