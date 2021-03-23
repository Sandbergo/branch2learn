import ecole
import os
import gzip
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Callable
from utilities.general import Logger
from tqdm import tqdm


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


def generate_instances(problem_type: str, num_samples: int, path: str, log: Callable) -> None:

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(
        observation_function=(ExploreThenStrongBranch(expert_probability=0.2),
                              ecole.observation.NodeBipartite()),
        scip_params=scip_parameters)

    env.seed(0)


    log(f'Generating {num_samples} {problem_type} instances')
    train_files = [str(path) for path in Path(
        f'branch2learn/data/instances/{problem_type}/train'
        ).glob('instance_*.lp')]

    episode_counter, sample_counter = 0, 0

    # We will solve problems (run episodes) until we have saved enough samples
    for _file in train_files:
        episode_counter += 1
        observation, action_set, _, done, _ = env.reset(_file)
        while not done:
            (scores, scores_are_expert), node_observation = observation

            node_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices,
                                node_observation.edge_features.values),
                                node_observation.column_features)

            action = action_set[scores[action_set].argmax()]

            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert:

                sample_counter += 1

                data = [node_observation, action, action_set, scores]
                filename = f'{path}/sample_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

            observation, action_set, _, done, _ = env.step(action)

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

    basedir = f'branch2learn/data/samples/{PROBLEM_TYPE}/'

    train_path = f'{basedir}/train/'
    os.makedirs(Path(train_path), exist_ok=True)
    valid_path = f'{basedir}/valid/'
    os.makedirs(Path(valid_path), exist_ok=True)
    test_path = f'{basedir}/test/'
    os.makedirs(Path(test_path), exist_ok=True)

    log('Generating training files')
    generate_instances(problem_type=PROBLEM_TYPE, num_samples=0, path=train_path, log=log)
    log('Generating valid files')
    generate_instances(problem_type=PROBLEM_TYPE, num_samples=0, path=valid_path, log=log)
    log('Generating test files')
    generate_instances(problem_type=PROBLEM_TYPE, num_samples=0, path=test_path, log=log)

    log('End of data generation.')
