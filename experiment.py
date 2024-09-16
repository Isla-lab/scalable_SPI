import gzip
import pickle
import numpy as np
import ast

import pandas as pd

from batch_rl_algorithms.mcts_spibb import MCTSSPIBB
from batch_rl_algorithms.sdp_spibb import SDPSPIBB
from batch_rl_algorithms.sdp_spibb_gen import SDPSPIBBGen
from batch_rl_algorithms.spibb import spibb, policy_evaluation_exact
from batch_rl_algorithms.spibb_gen import SPIBBGen
from envs.baseline_policy import SysAdminGenerativeBaselinePolicy, TaxiAbsorbingBaselinePolicy, \
    ScalableTaxiAbsorbingBaselinePolicy
from envs.sysadmin import SysAdmin
from envs.taxi_absorbing import TaxiAbsorbing
from envs.taxi_absorbing_gen import TaxiAbsorbingGen
from experiments.Experiments.Safety.SysAdmin.Utility.garnets import Garnets
from experiments.Experiments.Safety.SysAdmin.Utility.modelTransitions import ModelTransitions
from experiments.Experiments.Safety.SysAdmin.Utility.spibb_utils import compute_mask_N_wedge, generate_batch

# Translate the names from the algorithms to the class.
algorithm_name_dict = {SDPSPIBB.NAME: SDPSPIBB, SDPSPIBBGen.NAME: SDPSPIBBGen, spibb.NAME: spibb, MCTSSPIBB.NAME: MCTSSPIBB}


class Experiment:

    def __init__(self, experiment_config, seed, grid, machine_specific_experiment_directory):
        """
        :param experiment_config: config file which describes the experiment
        :param seed: seed for this experiment
        :param nb_iterations: number of iterations of this experiment
        :param machine_specific_experiment_directory: the directory in which the results will be stored
        """
        self.seed = seed
        np.random.seed(seed)
        self.experiment_config = experiment_config
        self.machine_specific_experiment_directory = machine_specific_experiment_directory
        self.algorithms_dict = ast.literal_eval(self.experiment_config['ALGORITHMS']['algorithms_dict'])
        self.filename_header = f'results_{seed}'
        self.filename_header = f'results_{seed}'
        print(f'Initialising experiment with seed {seed}')
        print(f'The machine_specific_experiment_directory is {self.machine_specific_experiment_directory}.')
        self._set_env_params()

    def run(self):
        """
        Runs the experiment.
        """
        pass

    def _set_env_params(self):
        pass

    def _run_algorithms(self):
        """
        Runs all algorithms for one data set.
        """
        pass


class SysAdminExperiment_Safety(Experiment):
    # Inherits from the base class Experiment to implement the SysAdmin experiment specifically.

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the SysAdmin experiment.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_5", "nb_traj_500", "nb_traj_5000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)

    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/SysAdmin/'
        self.real_T = np.load(path + 'T.npy')
        # garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        # garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        # optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        # optimal.fit()

        self.pi_b = np.load(path + 'Baseline_policy.npy')
        # values_baseline = policy_evaluation_exact(self.pi_b, self.R, self.real_T, self.gamma)[0][0]
        # values_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        # perf_baseline = [values_baseline] * self.n_iterations
        # perf_optimal = [values_optimal] * self.n_iterations

        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_baseline.npz',
        #                     perf_baseline)
        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_optimal.npz',
        #                     perf_optimal)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions, batch_traj = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b)
                # model = ModelTransitions(batch_traj, self.n_states, self.n_actions)
                # mask, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj)

                # self.save_object(model.transitions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def create_known_states_actions(self, iteration, number_trajectory):
        file = open(
            f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/batch_traj.txt',
            'r')
        batch = eval(file.read())
        known_states_actions = dict()
        for tr in batch:
            state = tr[1]
            action = tr[0]

            if state not in known_states_actions.keys():
                known_states_actions[state] = []

            if action not in known_states_actions[state]:
                known_states_actions[state].append(action)

        self.save_object(known_states_actions,
                         f'experiments/Experiments/Safety/SysAdmin/Results/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SPIBBGen.NAME}:
                self._run_spibb(iteration, number_trajectory)
            elif key in {SDPSPIBB.NAME, SDPSPIBB}:
                self.create_known_states_actions(iteration, number_trajectory)
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {spibb.NAME, spibb}:
                self._run_spibb(iteration, number_trajectory)

    def _run_spibb(self, iteration, number_trajectory):

        MLE_T, mask, _ = self.load_unfact_model(
            f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/it_{iteration}/Dsize_{number_trajectory}/model.pkl',
            f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/mask.npy',
            None)

        spibb_policy = spibb(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R, 'Pi_b_SPIBB')
        spibb_policy.fit()

        # result = spibb_policy.policy_evaluation_exact(spibb_policy.get_distribution(), self.R, self.real_T, self.gamma)
        result = spibb_policy.policy_evaluation_exact(spibb_policy.pi, self.R, self.real_T, self.gamma)

        # df = pd.DataFrame()
        # df['spibb'] = result[0]
        #
        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

    def _run_sdp_spibb(self, iteration, number_trajectory):

        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/it_{iteration}/Dsize_{number_trajectory}/model.pkl',
            f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/mask.npy',
            f'experiments/Experiments/Safety/SysAdmin/Results/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        # result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.get_distribution(), self.R, self.real_T, self.gamma)
        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        # df = pd.read_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')
        # # df = pd.DataFrame()
        # df['SDPSPIBB'] = result[0]

        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):

        MLE_T = pickle.load(open(path_MLE_T, 'rb'))

        mask = np.load(path_mask)

        if path_known_state_actions is not None:
            with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
                known_state_actions = pickle.load(input_file)
        else:
            known_state_actions = None

        return MLE_T.transitions, mask, known_state_actions


class SysAdminExperiment_Scalability(Experiment):
    # Inherits from the base class Experiment to implement the SysAdmin experiment specifically for 4-12 machines.

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the SysAdmin experiment 4-12 machines.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_5000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)

    def run(self, ):
        """
        Runs the experiment.
        """
        path = f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/'
        self.real_T = np.load(path + 'T.npy')
        # garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        # garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        # optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        # optimal.fit()

        self.pi_b = np.load(path + 'Baseline_policy.npy')
        # values_baseline = policy_evaluation_exact(self.pi_b, self.R, self.real_T, self.gamma)[0][0]
        # values_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        # perf_baseline = [values_baseline] * self.n_iterations
        # perf_optimal = [values_optimal] * self.n_iterations

        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_baseline.npz',
        #                     perf_baseline)
        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_optimal.npz',
        #                     perf_optimal)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions, batch_traj = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b)
                # model = ModelTransitions(batch_traj, self.n_states, self.n_actions)
                # mask, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj)

                # self.save_object(model.transitions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def compute_mask_N_wedge(self, nb_states, nb_actions, N_wedge, batch):
        count_state_action = np.zeros((nb_states, nb_actions))
        for [action, state, next_state, reward] in batch:
            # for [action, state, next_state] in batch:
            count_state_action[state, action] += 1
        return count_state_action > N_wedge, count_state_action

    def create_known_states_actions(self, iteration, number_trajectory):
        file = open(
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/batch_traj.txt',
            'r')
        batch = eval(file.read())
        known_states_actions = dict()
        for tr in batch:
            state = tr[1]
            action = tr[0]

            if state not in known_states_actions.keys():
                known_states_actions[state] = []

            if action not in known_states_actions[state]:
                known_states_actions[state].append(action)

        self.save_object(known_states_actions,
                         f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SPIBBGen.NAME}:
                self._run_spibb(iteration, number_trajectory)
            elif key in {SDPSPIBB.NAME, SDPSPIBB}:
                self.create_known_states_actions(iteration, number_trajectory)
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {spibb.NAME, spibb}:
                self._run_spibb(iteration, number_trajectory)

    def _run_spibb(self, iteration, number_trajectory):

        MLE_T, mask, _ = self.load_unfact_model(
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/MLE_models/it_{iteration}/Dsize_{number_trajectory}/model.pkl',
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/mask.npy',
            None)

        spibb_policy = spibb(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R, 'Pi_b_SPIBB')
        spibb_policy.fit()

        # result = spibb_policy.policy_evaluation_exact(spibb_policy.get_distribution(), self.R, self.real_T, self.gamma)
        result = spibb_policy.policy_evaluation_exact(spibb_policy.pi, self.R, self.real_T, self.gamma)

        # df = pd.DataFrame()
        # df['spibb'] = result[0]
        #
        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/results.csv')

    def _run_sdp_spibb(self, iteration, number_trajectory):

        import time
        start = time.time()
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/MLE_models/it_{iteration}/Dsize_{number_trajectory}/model.pkl',
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/MLE_models/it_{iteration}/Dsize_{number_trajectory}/n_wedge_{self.n_wedge}/mask.npy',
            f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy,
                                                          self.R, self.real_T, self.gamma)[0][0]

        end = time.time()
        print(end - start)
        # df = pd.read_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')
        # # df = pd.DataFrame()
        # df['SDPSPIBB'] = result[0]

        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Experiments/Scalability/SysAdmin/Results/{self.n_machines}_machines/results.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):

        MLE_T = pickle.load(open(path_MLE_T, 'rb'))

        mask = np.load(path_mask)

        if path_known_state_actions is not None:
            with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
                known_state_actions = pickle.load(input_file)
        else:
            known_state_actions = None

        return MLE_T.transitions, mask, known_state_actions


class SysAdminExperiment_Scalability_Extended(Experiment):
    # Inherits from the base class Experiment to implement the SysAdmin experiment specifically for >= 13 machines.

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the SysAdmin experiment >= 13 machines.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.env = SysAdmin(self.n_machines)
        self.pi_b = SysAdminGenerativeBaselinePolicy(self.env, self.gamma)
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_5000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)

    def run(self, ):
        """
        Runs the experiment.
        """

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures
                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def compute_mask_N_wedge(self, nb_states, nb_actions, N_wedge, batch):
        count_state_action = np.zeros((nb_states, nb_actions))
        for [action, state, next_state, reward] in batch:
            # for [action, state, next_state] in batch:
            count_state_action[state, action] += 1
        return count_state_action > N_wedge, count_state_action

    def create_known_states_actions(self, iteration, number_trajectory):
        path = f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/Dataset/batch_SysAdmin_{self.n_machines}m_it_{iteration}.pkl'

        with open(path, "rb") as input_file:
            batch = pickle.load(input_file)

        initial_policy = dict()
        initial_qvalues = dict()
        known_states_actions = dict()
        # tr = [state, action, reward, next_state]
        for b in batch:
            for tr in b:
                state = tr[0]
                action = tr[1]

                if state not in initial_policy.keys():
                    initial_policy[state] = self.pi_b.generative_baseline(state, True)

                if state not in initial_qvalues.keys():
                    initial_qvalues[state] = np.zeros(self.n_actions)

                if state not in known_states_actions.keys():
                    known_states_actions[state] = []

                if action not in known_states_actions[state]:
                    known_states_actions[state].append(action)
                    known_states_actions[state].sort()

        self.save_object(known_states_actions,
                         f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        self.save_object(initial_policy,
                         f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/initial_policy.pickle')

        self.save_object(initial_qvalues,
                         f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/initial_qvalues.pickle')

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SDPSPIBBGen.NAME, SDPSPIBBGen}:
                self.create_known_states_actions(iteration, number_trajectory)
                self._run_sdp_spibb(iteration, number_trajectory)

    def _run_sdp_spibb(self, iteration, number_trajectory):

        import time
        start = time.time()
        MLE_T, mask, known_states_actions, initial_policy, initial_qvalues = self.load_unfact_model(
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/MLE_models/MLE_T_SysAdmin_{self.n_machines}m_it_{iteration}.pkl',
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/Masks/mask_SysAdmin_{self.n_machines}m_it_{iteration}.pkl',
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/initial_policy.pickle',
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/initial_qvalues.pickle')

        sdp_spibb_policy = SDPSPIBBGen(self.env, self.pi_b, self.n_states, self.n_actions, MLE_T, mask,
                                       known_states_actions, initial_policy, initial_qvalues, self.gamma)
        sdp_spibb_policy.fit()

        end = time.time()
        print(end - start)
        sdp_spibb_policy = sdp_spibb_policy.get_distribution_generative()
        list_discounted_sdp_spibb = []
        for run in range(20):
            print('Run: %s' % run)
            state = self.env.reset()
            array_of_visited_states = []
            array_of_selected_actions = []
            list_rewards_sdp_spibb = []
            for step in range(20):
                if state in MLE_T.keys():
                    action = np.argmax(sdp_spibb_policy[state])
                else:
                    action = self.pi_b.generative_baseline(state, False)

                array_of_visited_states.append(state)
                array_of_selected_actions.append(action)
                state, reward, _ = self.env.step(state, action)
                list_rewards_sdp_spibb.append(reward)

            discounted_return_mcts_spibb = 0
            for step in range(20):
                discounted_return_mcts_spibb += list_rewards_sdp_spibb[step] * pow(self.gamma, step)

            list_discounted_sdp_spibb.append(discounted_return_mcts_spibb)

        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = np.mean(list_discounted_sdp_spibb)
        self.df.to_csv(
            f'experiments/Experiments/Scalability_Extended/SysAdmin/Results/{self.n_machines}_machines/results.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions, path_initial_policy,
                          path_initial_qvalues):

        with open(path_MLE_T, "rb") as input_file:
            MLE_T = pickle.load(input_file)

        with open(path_mask, "rb") as input_file:
            mask = pickle.load(input_file)

        with gzip.open(path_known_state_actions, "rb") as input_file:
            known_state_actions = pickle.load(input_file)

        with gzip.open(path_initial_policy, "rb") as input_file:
            initial_policy = pickle.load(input_file)

        with gzip.open(path_initial_qvalues, "rb") as input_file:
            initial_qvalues = pickle.load(input_file)

        return MLE_T, mask, known_state_actions, initial_policy, initial_qvalues


class GridworldExperiment_Safety(Experiment):
    # Inherits from the base class Experiment to implement the Gridworld experiment specifically.

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the Gridworld experiment.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.gridsize = int(self.experiment_config['ENV_PARAMETERS']['gridsize'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = self.gridsize * self.gridsize
        self.n_actions = 4
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_2", "nb_traj_10", "nb_traj_100", "nb_traj_1000", "nb_traj_10000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)

    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/Gridworld/'
        self.real_T = np.load(path + 'T.npy')
        # garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        # garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        # optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        # optimal.fit()

        self.pi_b = pickle.load(open(path + 'Baseline_policy.pkl', 'rb'))

        # values_baseline = policy_evaluation_exact(self.pi_b, self.R, self.real_T, self.gamma)[0][0]
        # values_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        # perf_baseline = [values_baseline] * self.n_iterations
        # perf_optimal = [values_optimal] * self.n_iterations

        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_baseline.npz',
        #                     perf_baseline)
        # np.savez_compressed('experiments/Experiments/Safety/SysAdmin/Results/Performance_opt_baseline/perf_optimal.npz',
        #                     perf_optimal)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions, batch_traj = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b)
                # model = ModelTransitions(batch_traj, self.n_states, self.n_actions)
                # mask, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj)

                # self.save_object(model.transitions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions,
                #                  f'experiments/Experiments/Safety/SysAdmin/Results/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def create_known_states_actions(self, iteration, number_trajectory):
        file = open(
            f'experiments/Experiments/Safety/Gridworld/Results/MLE_models/it_{iteration}/dsize_{number_trajectory}/MLE/batch_traj.txt',
            'r')
        batch = eval(file.read())
        known_states_actions = dict()
        for tr in batch:
            state = tr[1]
            action = tr[0]

            if state not in known_states_actions.keys():
                known_states_actions[state] = []

            if action not in known_states_actions[state]:
                known_states_actions[state].append(action)

        self.save_object(known_states_actions,
                         f'experiments/Experiments/Safety/Gridworld/Results/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SPIBBGen.NAME}:
                self._run_spibb(iteration, number_trajectory)
            elif key in {SDPSPIBB.NAME, SDPSPIBB}:
                self.create_known_states_actions(iteration, number_trajectory)
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {spibb.NAME, spibb}:
                self._run_spibb(iteration, number_trajectory)

    def _run_spibb(self, iteration, number_trajectory):

        MLE_T, mask, _ = self.load_unfact_model(
            f'experiments/Experiments/Safety/Gridworld/Results/MLE_models/it_{iteration}/dsize_{number_trajectory}/MLE/model.pkl',
            f'experiments/Experiments/Safety/Gridworld/Results/MLE_models/it_{iteration}/dsize_{number_trajectory}/MLE/nwedge_{self.n_wedge}/mask.npy',
            None)

        spibb_policy = spibb(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R, 'Pi_b_SPIBB')
        spibb_policy.fit()

        # result = spibb_policy.policy_evaluation_exact(spibb_policy.get_distribution(), self.R, self.real_T, self.gamma)
        result = spibb_policy.policy_evaluation_exact(spibb_policy.pi, self.R, self.real_T, self.gamma)

        # df = pd.DataFrame()
        # df['spibb'] = result[0]
        #
        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv('experiments/Experiments/Safety/Gridworld/Results/results.csv')

    def _run_sdp_spibb(self, iteration, number_trajectory):

        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Experiments/Safety/Gridworld/Results/MLE_models/it_{iteration}/dsize_{number_trajectory}/MLE/model.pkl',
            f'experiments/Experiments/Safety/Gridworld/Results/MLE_models/it_{iteration}/dsize_{number_trajectory}/MLE/nwedge_{self.n_wedge}/mask.npy',
            f'experiments/Experiments/Safety/Gridworld/Results/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        # result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.get_distribution(), self.R, self.real_T, self.gamma)
        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        # df = pd.read_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')
        # # df = pd.DataFrame()
        # df['SDPSPIBB'] = result[0]

        # df.to_csv('experiments/Experiments/Safety/SysAdmin/Results/results.csv')

        print(f"Results:{result}")
        self.df.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df.to_csv('experiments/Experiments/Safety/Gridworld/Results/results.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):

        MLE_T = pickle.load(open(path_MLE_T, 'rb'))

        mask = np.load(path_mask)

        if path_known_state_actions is not None:
            with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
                known_state_actions = pickle.load(input_file)
        else:
            known_state_actions = None

        return MLE_T.transitions, mask, known_state_actions

class Ablation_study_01(Experiment):
    # Inherits from the base class Experiment to implement the Ablation study 01 on SysAdmin, varying the baseline policy

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the Ablation study 01 on SysAdmin, varying the baseline policy.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_5000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df_rand = pd.DataFrame(data, columns=columns)
        self.df_sub_opt = pd.DataFrame(data, columns=columns)


    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/SysAdmin/'
        self.real_T = np.load(path + 'T.npy')
        garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        optimal.fit()

        pi_rand = np.ones((self.n_states, self.n_actions)) / self.n_actions
        rho = 0.9
        self.pi_b_sub_optimal = rho * optimal.pi + (1-rho) * pi_rand
        self.pi_b_random = pi_rand

        #
        # values_pi_b_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        # values_pi_b_sub_optimal = policy_evaluation_exact(self.pi_b_sub_optimal, self.R, self.real_T, self.gamma)[0][0]
        # values_pi_b_random = policy_evaluation_exact(self.pi_b_random, self.R, self.real_T, self.gamma)[0][0]
        #
        # perf_pi_b_optimal = [values_pi_b_optimal] * self.n_iterations
        # perf_pi_b_sub_optimal = [values_pi_b_sub_optimal] * self.n_iterations
        # perf_pi_b_random = [values_pi_b_random] * self.n_iterations
        #
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_optimal.npz',
        #                     perf_pi_b_optimal)
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_sub_optimal.npz',
        #                     perf_pi_b_sub_optimal)
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_random.npz',
        #                     perf_pi_b_random)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions_sub_opt, batch_traj_sub_opt = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b_sub_optimal)
                # known_states_actions_rand, batch_traj_rand = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b_random)
                #
                # model_sub_opt = ModelTransitions(batch_traj_sub_opt, self.n_states, self.n_actions)
                # model_rand = ModelTransitions(batch_traj_rand, self.n_states, self.n_actions)
                #
                # mask_sub_opt, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj_sub_opt)
                # mask_rand, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj_rand)
                #
                # self.save_object(model_sub_opt.transitions,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask_sub_opt,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions_sub_opt,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')
                #
                # self.save_object(model_rand.transitions,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask_rand,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions_rand,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SDPSPIBB.NAME, SDPSPIBB}:
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {MCTSSPIBB.NAME, MCTSSPIBB}:
                self._run_mcts_spibb(iteration, number_trajectory)

    def _run_sdp_spibb(self, iteration, number_trajectory):
        print('Testing SDP-SPIBB with random baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_rand/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_random, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        print(f"Results:{result}")
        self.df_rand.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_rand.to_csv('experiments/Ablation_study/Varying_baselines/sdp_spibb_results_with_pib_rand.csv')

        print('Testing SDP-SPIBB with sub-optimal baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_sub_optimal, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        print(f"Results:{result}")
        self.df_sub_opt.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_sub_opt.to_csv('experiments/Ablation_study/Varying_baselines/sdp_spibb_results_with_pib_sub_opt.csv')

    def _run_mcts_spibb(self, iteration, number_trajectory):
        print('Testing MCTS-SPIBB with random baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_rand/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        q_values_baseline = np.zeros((self.n_states, self.n_actions))
        budget = 0
        # states_to_sim = list(range(self.n_states))
        states_to_sim = list(known_states_actions.keys())
        n_sim = 10000

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_random, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, exploration_costant=5, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df_rand.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_rand.to_csv(f'experiments/Ablation_study/Varying_baselines/mcts_spibb_results_with_pib_rand_{n_sim}_new.csv')

        print('Testing MCTS-SPIBB with sub-optimal baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_sub_optimal, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df_sub_opt.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_sub_opt.to_csv(f'experiments/Ablation_study/Varying_baselines/mcts_spibb_results_with_pib_sub_opt_{n_sim}_new.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)

        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)

        with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
            known_state_actions = pickle.load(input_file)

        return MLE_T, mask, known_state_actions


class Ablation_study_02(Experiment):
    # Inherits from the base class Experiment to implement the Ablation study 01 on SysAdmin, varying the number of simulations

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the Ablation study 01 on SysAdmin, varying the baseline policy.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ["nb_traj_5000"]
        data = np.zeros((self.n_iterations, len(columns)))
        self.df_rand = pd.DataFrame(data, columns=columns)
        self.df_sub_opt = pd.DataFrame(data, columns=columns)


    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/SysAdmin/'
        self.real_T = np.load(path + 'T.npy')
        garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        optimal.fit()

        pi_rand = np.ones((self.n_states, self.n_actions)) / self.n_actions
        rho = 0.9
        self.pi_b_sub_optimal = rho * optimal.pi + (1-rho) * pi_rand
        self.pi_b_random = pi_rand

        #
        # values_pi_b_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        # values_pi_b_sub_optimal = policy_evaluation_exact(self.pi_b_sub_optimal, self.R, self.real_T, self.gamma)[0][0]
        # values_pi_b_random = policy_evaluation_exact(self.pi_b_random, self.R, self.real_T, self.gamma)[0][0]
        #
        # perf_pi_b_optimal = [values_pi_b_optimal] * self.n_iterations
        # perf_pi_b_sub_optimal = [values_pi_b_sub_optimal] * self.n_iterations
        # perf_pi_b_random = [values_pi_b_random] * self.n_iterations
        #
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_optimal.npz',
        #                     perf_pi_b_optimal)
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_sub_optimal.npz',
        #                     perf_pi_b_sub_optimal)
        # np.savez_compressed('experiments/Ablation_study/Varying_baselines/perf_pi_b_random.npz',
        #                     perf_pi_b_random)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions_sub_opt, batch_traj_sub_opt = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b_sub_optimal)
                # known_states_actions_rand, batch_traj_rand = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b_random)
                #
                # model_sub_opt = ModelTransitions(batch_traj_sub_opt, self.n_states, self.n_actions)
                # model_rand = ModelTransitions(batch_traj_rand, self.n_states, self.n_actions)
                #
                # mask_sub_opt, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj_sub_opt)
                # mask_rand, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj_rand)
                #
                # self.save_object(model_sub_opt.transitions,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask_sub_opt,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions_sub_opt,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')
                #
                # self.save_object(model_rand.transitions,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask_rand,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions_rand,
                #                  f'experiments/Ablation_study/Varying_baselines/Data_rand/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {MCTSSPIBB.NAME, MCTSSPIBB}:
                self._run_mcts_spibb(iteration, number_trajectory)

    def _run_mcts_spibb(self, iteration, number_trajectory):
        print('Testing MCTS-SPIBB with random baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_rand/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_rand/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        q_values_baseline = np.zeros((self.n_states, self.n_actions))
        budget = 0
        # states_to_sim = list(range(self.n_states))
        states_to_sim = list(known_states_actions.keys())
        n_sim = 100

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_random, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, exploration_costant=5, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df_rand.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_rand.to_csv(f'experiments/Ablation_study/Varying_nsims/mcts_spibb_results_with_pib_rand_{n_sim}.csv')

        print('Testing MCTS-SPIBB with sub-optimal baseline policy')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_baselines/Data_sub_opt/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b_sub_optimal, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df_sub_opt.loc[iteration, f"nb_traj_{number_trajectory}"] = result
        self.df_sub_opt.to_csv(f'experiments/Ablation_study/Varying_nsims/mcts_spibb_results_with_pib_sub_opt_{n_sim}.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)

        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)

        with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
            known_state_actions = pickle.load(input_file)

        return MLE_T, mask, known_state_actions

class Ablation_study_03(Experiment):
    # Inherits from the base class Experiment to implement the Ablation study 03 on SysAdmin, varying the dataset size

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the Ablation study 03 on SysAdmin, varying the dataset size.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ['5', '1000', '5000', '10000', '100000']
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)


    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/SysAdmin/'
        self.real_T = np.load(path + 'T.npy')
        garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        optimal.fit()

        pi_rand = np.ones((self.n_states, self.n_actions)) / self.n_actions
        rho = 0.7
        self.pi_b = rho * optimal.pi + (1-rho) * pi_rand

        values_pi_b_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        values_pi_b = policy_evaluation_exact(self.pi_b, self.R, self.real_T, self.gamma)[0][0]

        perf_pi_b_optimal = [values_pi_b_optimal] * self.n_iterations
        perf_pi_b = [values_pi_b] * self.n_iterations

        np.savez_compressed('experiments/Ablation_study/Varying_dataset_size/perf_pi_b_optimal.npz',
                            perf_pi_b_optimal)
        np.savez_compressed('experiments/Ablation_study/Varying_dataset_size/perf_pi_b.npz',
                            perf_pi_b)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions, batch_traj = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b)
                # model = ModelTransitions(batch_traj, self.n_states, self.n_actions)
                #
                # mask, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj)
                #
                # self.save_object(model.transitions,
                #                  f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask,
                #                  f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions,
                #                  f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SDPSPIBB.NAME, SDPSPIBB}:
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {MCTSSPIBB.NAME, MCTSSPIBB}:
                self._run_mcts_spibb(iteration, number_trajectory)

    def _run_sdp_spibb(self, iteration, number_trajectory):
        print('Testing SDP-SPIBB')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        print(f"Results:{result}")
        self.df.loc[iteration, f"{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Ablation_study/Varying_dataset_size/sdp_spibb_results_nb_traj_{number_trajectory}.csv')

    def _run_mcts_spibb(self, iteration, number_trajectory):
        print('Testing MCTS-SPIBB')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        q_values_baseline = np.zeros((self.n_states, self.n_actions))
        budget = 0
        # states_to_sim = list(range(self.n_states))
        states_to_sim = list(known_states_actions.keys())
        n_sim = 1000

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, exploration_costant=5, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df.loc[iteration, f"{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Ablation_study/Varying_dataset_size/mcts_spibb_results_nb_traj_{number_trajectory}.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)

        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)

        with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
            known_state_actions = pickle.load(input_file)

        return MLE_T, mask, known_state_actions

class Ablation_study_04(Experiment):
    # Inherits from the base class Experiment to implement the Ablation study 04 multi-agent SysAdmin

    def _set_env_params(self):
        """
        Reads in all parameters necessary to set up the Ablation study 03 on SysAdmin, varying the dataset size.
        """
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['gamma'])
        self.n_machines = int(self.experiment_config['ENV_PARAMETERS']['machines'])
        self.number_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['length_trajectory'])[0]
        self.n_states = 2 ** self.n_machines
        self.n_actions = self.n_machines + 1
        self.n_iterations = int(self.experiment_config['ENV_PARAMETERS']['n_iterations'])
        self.n_wedge = ast.literal_eval(self.experiment_config['ENV_PARAMETERS']['n_wedge'])
        self.n_steps = int(self.experiment_config['ENV_PARAMETERS']['n_steps'])
        self.results = pd.DataFrame()
        columns = ['5', '1000', '5000', '10000', '100000']
        data = np.zeros((self.n_iterations, len(columns)))
        self.df = pd.DataFrame(data, columns=columns)


    def run(self, ):
        """
        Runs the experiment.
        """
        path = 'experiments/Experiments/Safety/SysAdmin/'
        self.real_T = np.load(path + 'T.npy')
        garnet = Garnets(self.n_states, self.n_actions, 1, self_transitions=0)
        garnet.transition_function = self.real_T
        self.R = np.load(path + 'R.npy')
        mask_0 = np.full((self.n_states, self.n_actions), True)
        optimal = spibb(self.gamma, self.n_states, self.n_actions, mask_0, mask_0, self.real_T, self.R, 'default')
        optimal.fit()

        pi_rand = np.ones((self.n_states, self.n_actions)) / self.n_actions
        rho = 0.7
        self.pi_b = rho * optimal.pi + (1-rho) * pi_rand

        values_pi_b_optimal = policy_evaluation_exact(optimal.pi, self.R, self.real_T, self.gamma)[0][0]
        values_pi_b = policy_evaluation_exact(self.pi_b, self.R, self.real_T, self.gamma)[0][0]

        perf_pi_b_optimal = [values_pi_b_optimal] * self.n_iterations
        perf_pi_b = [values_pi_b] * self.n_iterations

        np.savez_compressed('experiments/Ablation_study/Varying_dataset_size/perf_pi_b_optimal.npz',
                            perf_pi_b_optimal)
        np.savez_compressed('experiments/Ablation_study/Varying_dataset_size/perf_pi_b.npz',
                            perf_pi_b)

        for nb_trajectories in self.number_trajectory:
            for it in range(self.n_iterations):
                print(f'Iteration: {it}. Number of trajectories: {nb_trajectories} out of {self.number_trajectory}.')
                # N.B. We use old data structures

                # known_states_actions, batch_traj = generate_batch(nb_trajectories, self.length_trajectory, self.real_T, self.R, self.pi_b)
                # model = ModelTransitions(batch_traj, self.n_states, self.n_actions)
                #
                # mask, _ = compute_mask_N_wedge(self.n_states, self.n_actions, self.n_wedge, batch_traj)
                #
                # self.save_object(model.transitions,
                #                  f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(mask,
                #                  f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{it}_nb_traj_{nb_trajectories}.pickle')
                # self.save_object(known_states_actions,
                #                  f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{it}_nb_traj_{nb_trajectories}.pickle')

                self._run_algorithms(it, nb_trajectories)

    def save_object(self, obj, filename):
        with gzip.open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)

    def _run_algorithms(self, iteration, number_trajectory):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {SDPSPIBB.NAME, SDPSPIBB}:
                self._run_sdp_spibb(iteration, number_trajectory)
            elif key in {MCTSSPIBB.NAME, MCTSSPIBB}:
                self._run_mcts_spibb(iteration, number_trajectory)

    def _run_sdp_spibb(self, iteration, number_trajectory):
        print('Testing SDP-SPIBB')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        sdp_spibb_policy = SDPSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, mask, MLE_T, self.R,
                                    known_states_actions)
        sdp_spibb_policy.fit()

        result = sdp_spibb_policy.policy_evaluation_exact(sdp_spibb_policy.policy, self.R, self.real_T, self.gamma)[0][
            0]

        print(f"Results:{result}")
        self.df.loc[iteration, f"{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Ablation_study/Varying_dataset_size/sdp_spibb_results_nb_traj_{number_trajectory}.csv')

    def _run_mcts_spibb(self, iteration, number_trajectory):
        print('Testing MCTS-SPIBB')
        MLE_T, mask, known_states_actions = self.load_unfact_model(
            f'experiments/Ablation_study/Varying_dataset_size/MLE_models/MLE_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Masks/mask_it_{iteration}_nb_traj_{number_trajectory}.pickle',
            f'experiments/Ablation_study/Varying_dataset_size/Known_states_actions/known_states_actions_it_{iteration}_nb_traj_{number_trajectory}.pickle')

        q_values_baseline = np.zeros((self.n_states, self.n_actions))
        budget = 0
        # states_to_sim = list(range(self.n_states))
        states_to_sim = list(known_states_actions.keys())
        n_sim = 1000

        mcts_spibb = MCTSSPIBB(self.gamma, self.n_states, self.n_actions, self.pi_b, MLE_T, self.R,
                          mask, q_values_baseline, budget, self.seed, n_sims=n_sim, exploration_costant=5, states_to_sim=states_to_sim)

        mcts_spibb.fit()

        result = policy_evaluation_exact(mcts_spibb.pi, self.R, self.real_T, self.gamma)[0][0]

        print(f"Results:{result}")
        self.df.loc[iteration, f"{number_trajectory}"] = result
        self.df.to_csv(f'experiments/Ablation_study/Varying_dataset_size/mcts_spibb_results_nb_traj_{number_trajectory}.csv')

    def load_unfact_model(self, path_MLE_T, path_mask, path_known_state_actions):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)

        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)

        with gzip.open(f"{path_known_state_actions}", "rb") as input_file:
            known_state_actions = pickle.load(input_file)

        return MLE_T, mask, known_state_actions