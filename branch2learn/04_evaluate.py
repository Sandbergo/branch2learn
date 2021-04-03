import numpy as np
import torch
import torch_geometric
import os
import ecole
import argparse
from pathlib import Path

from utilities.general import Logger
from utilities.model import process
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='Model name.',
        choices=['gnn1', 'gnn2', 'mlp1', 'mlp2', 'mlp3'],
    )
    parser.add_argument(
        '-p', '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    POLICY_DICT = {'mlp1': MLP1Policy(), 'mlp2': MLP2Policy(), 'mlp3': MLP3Policy(),
                   'gnn1': GNN1Policy(), 'gnn2': GNN2Policy(), }
    PROBLEM = args.problem

    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        DEVICE = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = POLICY_DICT[args.model].to(DEVICE)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    Path('branch2learn/log/').mkdir(exist_ok=True)
    log = Logger(filename='branch2learn/log/04_evaluate')

    log(f'Model:   {args.model}')
    log(f'Problem: {PROBLEM}')
    log(f'Device:  {DEVICE}')
    log(str(policy))


    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 100} # TODO: revert to 2700
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generators = {
        'setcover':(
            ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000, density=0.05),
            ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=1000, density=0.05)
        ),
        'indset': (
            ecole.instance.IndependentSetGenerator(n_nodes=750, graph_type="barabasi_albert", affinity=4),
            ecole.instance.IndependentSetGenerator(n_nodes=1000, graph_type="barabasi_albert", affinity=4),
            ecole.instance.IndependentSetGenerator(n_nodes=1500, graph_type="barabasi_albert", affinity=4),

        ),
        'facilitites': (
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100, continuous_assignment=True),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=200, continuous_assignment=True),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=400, continuous_assignment=True),
        )
        'cauctions': (
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1000),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1500)
        ),
        }
    sizes = ['small', 'medium', 'large']
    for problem_type in generators.keys():
        node_list,time_list = [],[]
        completed = 0
        for i, instances in enumerate(generators[problem_type]):
            log(f'------ {problem_type}, {sizes[i]} ------')
            for instance_count, instance in zip(range(5), instances):

                observation, action_set, _, done, info = env.reset(instance)

                while not done:
                    with torch.no_grad():
                        observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(DEVICE),
                                    torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(DEVICE),
                                    torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(DEVICE),
                                    torch.from_numpy(observation.column_features.astype(np.float32)).to(DEVICE))
                        logits = policy(*observation)
                        action = action_set[logits[action_set.astype(np.int64)].argmax()]
                        observation, action_set, _, done, info = env.step(action)
                nb_nodes, time = info['nb_nodes'], info['time']
                node_list.append(nb_nodes)
                time_list.append(time)
                completed += int(done)

            log(f" nb nodes    {int(np.mean(node_list)): >4d}  |  time {np.mean(time_list): >6.2f} | completed   {completed : >2d}")
    
    log('End of evaluation.')
