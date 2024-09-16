# Scaling Safe Policy Improvement: Monte Carlo Tree Search and Policy Iteration Strategies

This project can be used to reproduce the experiments presented in:

- Federico Bianchi, Alberto Castellini, Edoardo Zorzi, Thiago D. Simão, Matthijs T. J. Spaan, Alessandro Farinelli. Scaling Safe Policy Improvement: Monte Carlo Tree Search and Policy Iteration Strategies


# Prerequisites

The project is implemented in Python 3.10 and tested on Ubuntu 22.04.2 LTS (for the full list of requirements please refer to file requirements.txt)

# Usage

We include the following:

    Libraries of the algorithms:
        - SPIBB (from Romain Laroche https://github.com/RomainLaroche/SPIBB)
        - MCTS-SPIBB (our method)
        - SDP-SPIBB (our method)

    Environments:
        Gridworld
        SysAdmin
        Multi-agent SysAdmin

1. In order to execute the code, set the path within the file paths.ini, then launch the file run_experiments.py setting the folder where the results will be stored and the seed, for example, Gridworld.ini gridworld_results 1234


2. To launch the experiments regarding safety in the Gridworld domain:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/Gridworld_safety.ini"

3. To launch the experiments regarding safety in the SysAdmin domain:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/SysAdmin_safety.ini"

4. To launch the experiments regarding scalability in the SysAdmin domain:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/SysAdmin_scalability.ini"
   or
   - Set the parameters (e.g., number of trajectories) within the file "experiments/SysAdmin_scalability_extended.ini"

5. To launch the experiments regarding varying the baseline policy:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/Ablation_study_01.ini"

6. To launch the experiments regarding the variation of the baseline policy type:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/Ablation_study_02.ini"

7. To launch the experiments regarding the variation of the dataset size:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/Ablation_study_03.ini"

8. To launch the experiments regarding the scalability on very large domain:
   - Set the parameters (e.g., number of trajectories) within the file "experiments/Ablation_study_04.ini"

# Contributors
- Federico Bianchi (federico.bianchi@univr.it or federicobianchi501@gmail.com)


# License

This project is GPLv3-licensed. 
