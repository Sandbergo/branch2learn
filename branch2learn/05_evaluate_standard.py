import numpy as np
import torch
import torch_geometric
import os
import argparse
import ecole
from pathlib import Path

from utilities.general import Logger
from utilities.model import process
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--brancher',
        help='Brancher name.',
        choices=['pc', 'fsb'],
    )
    parser.add_argument(
        '-p', '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    PROBLEM = args.problem
    POLICY_DICT = {'fsb': ecole.observation.StrongBranchingScores(), 
                   'pc': ecole.observation.Pseudocosts()}

    Path('branch2learn/log/').mkdir(exist_ok=True)
    log = Logger(filename='branch2learn/log/05_evaluate_standard')

    log(f'Policy:   {args.brancher}')
    log(f'Problem: {PROBLEM}')
    policy = POLICY_DICT[args.brancher]

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 100} # TODO: revert to 2700

    default_env = ecole.environment.Branching(
        observation_function=policy,
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generators = {
        'setcover':(
            ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=1000, density=0.05)
        ),
        'cauctions': (
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1000),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1500)
        ),

        'indset': (
            ecole.instance.IndependentSetGenerator(n_nodes=750, graph_type="erdos_renyi"),
            ecole.instance.IndependentSetGenerator(n_nodes=1000, graph_type="erdos_renyi"),
            ecole.instance.IndependentSetGenerator(n_nodes=1500, graph_type="erdos_renyi"),

        ),
        'facilitites': (
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=200),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=400),
        )
        }
    sizes = ['small', 'medium', 'large']
    for problem_type in generators.keys():
        for i, instances in enumerate(generators[problem_type]):
            node_list,time_list = [],[]
            completed = 0
            log(f'------ {problem_type}, {sizes[i]} ------')
            for instance_count, instance in zip(range(5), instances):

                observation, action_set, _, done, info = default_env.reset(instance)

                while not done:
                    scores = observation
                    action = action_set[scores[action_set].argmax()]
                    observation, action_set, _, done, info = default_env.step(action)
                
                nb_nodes, time = info['nb_nodes'], info['time']
                node_list.append(nb_nodes)
                time_list.append(time)
                completed += int(done)

            log(f" nb nodes    {int(np.mean(node_list)): >4d}  |  time {np.mean(time_list): >6.2f} | completed   {int(completed) : >2d}")
    
    log('End of evaluation.')
