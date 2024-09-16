from gym.envs.toy_text.discrete import categorical_sample
from gym.envs.toy_text.taxi import TaxiEnv, MAP
import numpy as np
from gym.envs.toy_text import discrete


class TaxiAbsorbing(TaxiEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations.

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger

    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.


    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.nb_states = 500
        self.nb_actions = 6
        self.number_of_rows = 5
        self.number_of_columns = 5
        self.maxRow = self.number_of_rows - 1
        self.maxColumn = self.number_of_columns - 1
        self.locs = [(0, 0), (0, self.maxColumn), (self.maxRow, 0), (self.maxRow, self.maxColumn)]
        self.taxi_row = 0
        self.taxi_col = 1
        self.pass_loc = 3
        self.dest_idx = 2
        initial_state_distrib = np.zeros(self.nb_states)

        P = {s: {a: [] for a in range(self.nb_actions)} for s in range(self.nb_states)}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            initial_state_distrib[state] += 1
                        for a in range(self.nb_actions):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a == 0:
                                newrow = min(row + 1, self.maxRow)
                            elif a == 1:
                                newrow = max(row - 1, 0)
                            # if a == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                            #     newcol = min(col + 1, self.maxColumn)
                            # elif a == 3 and self.desc[1 + row, 2 * col] == b":":
                            #     newcol = max(col - 1, 0)
                            if a == 2:
                                newcol = min(col + 1, self.maxColumn)
                            elif a == 3:
                                newcol = max(col - 1, 0)
                            elif a == 4:  # pickup
                                if (passidx < 4 and taxiloc == self.locs[passidx]):
                                    newpassidx = 4
                                else:
                                    reward = -10
                            elif a == 5:  # dropoff
                                if (taxiloc == self.locs[destidx]) and passidx == 4:
                                    newpassidx = destidx
                                    done = True
                                    reward = 20
                                elif (taxiloc in self.locs) and passidx == 4:
                                    newpassidx = self.locs.index(taxiloc)
                                else:
                                    reward = -10
                            if destidx == passidx:
                                # no more reward after the end of the episode
                                reward = 0
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(self, self.nb_states, self.nb_actions, P, initial_state_distrib)

        self.T = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.R = np.zeros((self.nb_states, self.nb_actions))

        for s in range(0, self.nb_states):
            for a in range(0, self.nb_actions):
                for elem in P[s][a]:
                    self.T[s][a][elem[1]] = elem[0]
                    self.R[s][a] = elem[2]

    def reward_function(self, state, action):
        return self.R[int(state)][int(action)]

    def reset(self):
        self.s = self.encode(self.taxi_row, self.taxi_col, self.pass_loc, self.dest_idx)
        self.lastaction = None
        return int(self.s)

    def reset(self):
        self.s = self.encode(self.taxi_row, self.taxi_col, self.pass_loc, self.dest_idx)
        self.lastaction = None
        return int(self.s)

    def step(self, s, a):
        transitions = self.P[s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        # self.s = s
        # self.lastaction = a
        return (int(s), r, d, {"prob": p})

    # def step(self, state, action):
    #     return self.T[state][action]
