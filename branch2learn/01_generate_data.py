"""
Data generation script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import os
import gzip
import pickle
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import ecole

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
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


def generate_instances(
    problem_type: str, num_samples: int, path: str, log: Callable
) -> None:

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }

    env = ecole.environment.Branching(
        observation_function=(
            ExploreThenStrongBranch(expert_probability=0.05),
            ecole.observation.NodeBipartite(),
        ),
        scip_params=scip_parameters,
    )

    env.seed(0)

    generators = {
        "setcover": ecole.instance.SetCoverGenerator(
            n_rows=500, n_cols=1000, density=0.05
        ),
        "cauctions": ecole.instance.CombinatorialAuctionGenerator(
            n_items=100, n_bids=500, add_item_prob=0.7
        ),
        "indset": ecole.instance.IndependentSetGenerator(
            n_nodes=500, graph_type="barabasi_albert", affinity=4
        ),
        "facilities": ecole.instance.CapacitatedFacilityLocationGenerator(
            n_customers=100, n_facilities=100
        ),
    }

    log(f"Generating {num_samples} {problem_type} instances")
    log(f"Saving samples in {path}")
    instances = generators[problem_type]

    episode_counter, sample_counter = 0, 0

    # We will solve problems (run episodes) until we have saved enough samples
    max_samples_reached = False
    while not max_samples_reached:
        episode_counter += 1

        observation, action_set, _, done, _ = env.reset(next(instances))
        while not done:
            (scores, scores_are_expert), node_observation = observation

            node_observation = (
                node_observation.row_features,
                (
                    node_observation.edge_features.indices,
                    node_observation.edge_features.values,
                ),
                node_observation.column_features,
            )

            action = action_set[scores[action_set].argmax()]

            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and not max_samples_reached:

                sample_counter += 1

                data = [node_observation, action, action_set, scores]
                filename = f"{path}/sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as out_file:
                    pickle.dump(data, out_file)

                if sample_counter >= num_samples:
                    max_samples_reached = True

            observation, action_set, _, done, _ = env.step(action)

        if episode_counter % 5 == 0:
            log(
                f"Episode {episode_counter}, {sample_counter} / {num_samples} samples collected"
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--problem",
        help="MILP instance type to process.",
        choices=["setcover", "cauctions", "facilities", "indset"],
        default="setcover",
    )
    args = parser.parse_args()

    TRAIN_SIZE = 50_000
    VALID_SIZE = 10_000
    TEST_SIZE = 10_000
    PROBLEM_TYPE = args.problem

    Path("branch2learn/log/").mkdir(exist_ok=True)
    log = Logger(filename="branch2learn/log/01_generate_data")

    basedir = f"branch2learn/data/samples/{PROBLEM_TYPE}/"

    train_path = f"{basedir}/train/"
    os.makedirs(Path(train_path), exist_ok=True)
    valid_path = f"{basedir}/valid/"
    os.makedirs(Path(valid_path), exist_ok=True)
    test_path = f"{basedir}/test/"
    os.makedirs(Path(test_path), exist_ok=True)

    generate_instances(
        problem_type=PROBLEM_TYPE, num_samples=TRAIN_SIZE, path=train_path, log=log
    )
    generate_instances(
        problem_type=PROBLEM_TYPE, num_samples=VALID_SIZE, path=valid_path, log=log
    )
    generate_instances(
        problem_type=PROBLEM_TYPE, num_samples=TEST_SIZE, path=test_path, log=log
    )

    log("End of data generation.")
