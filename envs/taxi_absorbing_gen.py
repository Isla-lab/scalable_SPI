from gym.envs.toy_text.taxi import TaxiEnv, MAP
import numpy as np
from gym.envs.toy_text import discrete


class TaxiAbsorbingGen:
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

    def __init__(self, grid):
        # 25x25
        MAP = [
            "+-------------------------------------------------+",
            "|R: | : : : : : : : : : : : : : : : : : : : : : :G|",
            "| : | : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| : : : : : : : : : : : : : : : : : : : : : : : : |",
            "| | : : : : : : : : : : : : : : : : : : : : : | : |",
            "|Y| : : : : : : : : : : : : : : : : : : : : : |B: |",
            "+-------------------------------------------------+",
        ]

        # # 10x10
        # MAP = [
        #     "+-------------------+",
        #     "|R: | : : : : : : :G|",
        #     "| : | : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| : : : : : : : : : |",
        #     "| | : : : : : : | : |",
        #     "|Y| : : : : : : |B: |",
        #     "+-------------------+",
        # ]

        # # 5x5
        # MAP = [
        #     "+---------+",
        #     "|R: | : :G|",
        #     "| : | : : |",
        #     "| : : : : |",
        #     "| | : | : |",
        #     "|Y| : |B: |",
        #     "+---------+",
        # ]
        self.desc = np.asarray(MAP, dtype='c')
        self.number_of_rows = grid
        self.number_of_columns = grid
        self.locations_destinations = 4
        self.locations_passenger = 5
        self.nb_states = self.number_of_rows * self.number_of_columns * self.locations_passenger * \
                         self.locations_destinations
        self.nb_actions = 6
        self.maxRow = self.number_of_rows - 1
        self.maxColumn = self.number_of_columns - 1
        self.locs = [(0, 0), (0, self.maxColumn), (self.maxRow, 0), (self.maxRow, self.maxColumn)]
        self.taxi_row = 0
        self.taxi_col = 1
        self.pass_loc = 3
        self.dest_idx = 2
        self.old_numb_rows = 5
        self.old_finish_episode = 20
        self.finish_episode = (100 + self.number_of_rows * self.old_finish_episode)/self.old_numb_rows

    def step(self, state, action):
        state = int(state)
        action = int(action)
        row, col, passidx, destidx = self.decode(state)
        newrow, newcol, newpassidx = row, col, passidx
        taxiloc = (row, col)
        reward = -1
        done = False
        if action == 0:
            newrow = min(row + 1, self.maxRow)
        elif action == 1:
            newrow = max(row - 1, 0)
        # if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
        #     newcol = min(col + 1, self.maxColumn)
        # elif action == 3 and self.desc[1 + row, 2 * col] == b":":
        #     newcol = max(col - 1, 0)
        if action == 2:
            newcol = min(col + 1, self.maxColumn)
        elif action == 3:
            newcol = max(col - 1, 0)
        elif action == 4:  # pickup
            if (passidx < 4 and taxiloc == self.locs[passidx]):
                newpassidx = 4
            else:
                reward = -10
        elif action == 5:  # dropoff
            if (taxiloc == self.locs[destidx]) and passidx == 4:
                newpassidx = destidx
                done = True
                reward = self.finish_episode
            elif (taxiloc in self.locs) and passidx == 4:
                newpassidx = self.locs.index(taxiloc)
            else:
                reward = -10
        if destidx == passidx:
            # no more reward after the end of the episode
            reward = 0
        newstate = self.encode(newrow, newcol, newpassidx, destidx)

        return newstate, reward, done, None

    def reward_function(self, state, action):
        state = int(state)
        action = int(action)
        row, col, passidx, destidx = self.decode(state)
        newrow, newcol, newpassidx = row, col, passidx
        taxiloc = (row, col)
        reward = -1
        done = False
        if action == 0:
            newrow = min(row + 1, self.maxRow)
        elif action == 1:
            newrow = max(row - 1, 0)
        # if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
        #     newcol = min(col + 1, self.maxColumn)
        # elif action == 3 and self.desc[1 + row, 2 * col] == b":":
        #     newcol = max(col - 1, 0)
        if action == 2:
            newcol = min(col + 1, self.maxColumn)
        elif action == 3:
            newcol = max(col - 1, 0)
        elif action == 4:  # pickup
            if (passidx < 4 and taxiloc == self.locs[passidx]):
                newpassidx = 4
            else:
                reward = -10
        elif action == 5:  # dropoff
            if (taxiloc == self.locs[destidx]) and passidx == 4:
                newpassidx = destidx
                done = True
                reward = self.finish_episode
            elif (taxiloc in self.locs) and passidx == 4:
                newpassidx = self.locs.index(taxiloc)
            else:
                reward = -10
        if destidx == passidx:
            # no more reward after the end of the episode
            reward = 0
        newstate = self.encode(newrow, newcol, newpassidx, destidx)

        return reward

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        # (5) 5, 5, 4
        i = taxi_row
        i *= self.number_of_columns
        i += taxi_col
        i *= self.locations_passenger
        i += pass_loc
        i *= self.locations_destinations
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % self.locations_destinations)
        i = i // self.locations_destinations
        out.append(i % self.locations_passenger)
        i = i // self.locations_passenger
        out.append(i % self.number_of_columns)
        i = i // self.number_of_columns
        out.append(i)
        assert 0 <= i < self.number_of_rows
        return reversed(out)

    def reset(self):
        self.initial_state = self.encode(self.taxi_row, self.taxi_col, self.pass_loc, self.dest_idx)

        return self.initial_state