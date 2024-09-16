import os
import sys
import configparser

from experiment import SysAdminExperiment_Safety, GridworldExperiment_Safety, \
    SysAdminExperiment_Scalability, SysAdminExperiment_Scalability_Extended, \
    Ablation_study_01, Ablation_study_02, Ablation_study_03, Ablation_study_04

directory = os.path.dirname(os.path.expanduser(__file__))
sys.path.append(directory)

path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
results_directory_absolute = path_config['PATHS']['results_path']

config_name = sys.argv[1]
experiment_config = configparser.ConfigParser()
experiment_config.read(os.path.join(directory, 'experiments', config_name))
experiment_directory_relative = experiment_config['META']['experiment_path_relative']
environment = experiment_config['META']['env_name']
machine_specific_directory = sys.argv[2]

experiment_directory = os.path.join(results_directory_absolute, experiment_directory_relative)
machine_specific_experiment_directory = os.path.join(experiment_directory, machine_specific_directory)

if not os.path.isdir(results_directory_absolute):
    os.mkdir(results_directory_absolute)
if not os.path.isdir(experiment_directory):
    os.mkdir(experiment_directory)
if not os.path.isdir(machine_specific_experiment_directory):
    os.mkdir(machine_specific_experiment_directory)


def run_experiment(seed):
    if 'SysAdmin_safety' in config_name:
        experiment = SysAdminExperiment_Safety(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)

    elif 'SysAdmin_scalability_extended' in config_name:
        experiment = SysAdminExperiment_Scalability_Extended(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)

    elif 'SysAdmin_scalability' in config_name:
        experiment = SysAdminExperiment_Scalability(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)

    elif 'Gridworld_safety' in config_name:
        experiment = GridworldExperiment_Safety(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)

    elif 'Ablation_study_01' in config_name:
        experiment = Ablation_study_01(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)
    elif 'Ablation_study_02' in config_name:
        experiment = Ablation_study_02(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)
    elif 'Ablation_study_03' in config_name:
        experiment = Ablation_study_03(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)
    elif 'Ablation_study_04' in config_name:
        experiment = Ablation_study_03(experiment_config=experiment_config, seed=seed,
                                                         grid=0,
                                                         machine_specific_experiment_directory=
                                                         machine_specific_experiment_directory)

    experiment.run()


if __name__ == '__main__':
    seed = int(sys.argv[3])
    # f = open(os.path.join(machine_specific_experiment_directory, "Exp_description.txt"), "w+")
    # f.write(f"This benchmark is used for {nb_iterations} iterations.\n")
    # f.write(f'The seed used is {seed}.\n')
    # f.write(f'Experiment starts at {time.ctime()}.')
    # f.close()
    run_experiment(seed)
