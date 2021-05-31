"""
Instance generation script.

File adapted from https://github.com/ds4dm/learn2branch
by Lars Sandberg @Sandbergo
May 2021
"""

import ecole
import os
import gzip
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Callable

from utilities.general import Logger


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability: float):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment
        (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


def generate_instances(instance_path: str, sample_path: str, num_samples: int, log: Callable) -> None:

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(
        observation_function=(ExploreThenStrongBranch(expert_probability=0.05),
                              ecole.observation.NodeBipartite()),
        scip_params=scip_parameters)

    env.seed(0)
    
    instance_files = [str(path) for path in Path(instance_path).glob('instance_*.lp')]
    log(f'Generating from {len(instance_files)} instances')
    
    episode_counter, sample_counter = 0, 0  # TODO: temp

    # We will solve problems (run episodes) until we have saved enough samples
    while True:
        for _file in instance_files:
            episode_counter += 1
            observation, action_set, _, done, _ = env.reset(_file)
            while not done:
                (scores, scores_are_expert), node_observation = observation

                action = action_set[scores[action_set].argmax()]

                # Only save samples if they are coming from the expert (strong branching)
                if scores_are_expert:

                    sample_counter += 1
                    if sample_counter > num_samples:
                        return

                    node_observation = (node_observation.row_features,
                                    (node_observation.edge_features.indices,
                                    node_observation.edge_features.values),
                                    node_observation.column_features)

                    data = [node_observation, action, action_set, scores]
                    filename = f'{sample_path}/sample_{sample_counter}.pkl'

                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(data, f)

                observation, action_set, _, done, _ = env.step(action)
                
            if episode_counter % 10 == 0:
                log(f"Episode {episode_counter}, {sample_counter} samples collected")
            
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
        default='setcover'
    )
    args = parser.parse_args()

    PROBLEM_TYPE = args.problem

    Path('branch2learn/log/').mkdir(exist_ok=True)
    log = Logger(filename='branch2learn/log/01_generate_data')

    log(f'Problem: {PROBLEM_TYPE}')

    basedir_samp = f'branch2learn/data/samples/{PROBLEM_TYPE}/'
    basedir_inst = f'branch2learn/data/instances/{PROBLEM_TYPE}/'

    train_path_inst = f'{basedir_inst}/train/'
    valid_path_inst = f'{basedir_inst}/valid/'
    test_path_inst = f'{basedir_inst}/test/'

    train_path_samp = f'{basedir_samp}/train/'
    os.makedirs(Path(train_path_samp), exist_ok=True)
    valid_path_samp = f'{basedir_samp}/valid/'
    os.makedirs(Path(valid_path_samp), exist_ok=True)
    test_path_samp = f'{basedir_samp}/test/'
    os.makedirs(Path(test_path_samp), exist_ok=True)

    log('Generating training files')
    generate_instances(instance_path=train_path_inst, sample_path=train_path_samp, num_samples=50_000, log=log)
    log('Generating valid files')
    generate_instances(instance_path=valid_path_inst, sample_path=valid_path_samp, num_samples=10_000, log=log)
    log('Generating test files')
    generate_instances(instance_path=test_path_inst, sample_path=test_path_samp, num_samples=10_000, log=log)

    log('End of data generation.')
