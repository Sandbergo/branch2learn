import numpy as np
import torch
import torch_geometric
import os
import argparse
import ecole
from pathlib import Path

from utilities.general import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--brancher',
        help='Brancher name.',
        choices=['pc', 'fsb', 'rpc'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    POLICY_DICT = {'fsb': 'vanillafullstrong', 
                   'pc': 'pscost',
                   'rpc': 'relpscost'}
    TIME_LIMIT = 2700

    Path('branch2learn/log/').mkdir(exist_ok=True)
    log = Logger(filename='branch2learn/log/05_evaluate_standard')

    log(f'Policy:   {args.brancher}')
    policy = POLICY_DICT[args.brancher]

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': TIME_LIMIT, 
                       f'branching/{policy}/priority': 666666}
    
    env = ecole.environment.Configuring(
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    generators = {
        'cauctions': (
           [str(path) for path in Path(
           f'branch2learn/data/instances/cauctions/test').glob('instance_*.lp')],
        ),
        'setcover': (
            [str(path) for path in Path(
           f'branch2learn/data/instances/setcover/test').glob('instance_*.lp')],
        ),
        }
    """'facilitites': (
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=200),
            ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=400),
        ),
        'indset': (
            ecole.instance.IndependentSetGenerator(n_nodes=750, graph_type="barabasi_albert", affinity=4),
            ecole.instance.IndependentSetGenerator(n_nodes=1000, graph_type="barabasi_albert", affinity=4),
            ecole.instance.IndependentSetGenerator(n_nodes=1500, graph_type="barabasi_albert", affinity=4),

        ),
        'cauctions': (
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1000),
            ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=1500)
        ),"""
    sizes = ['small', 'medium', 'large']
    for problem_type in generators.keys():
        for i, instances in enumerate(generators[problem_type]):
            if i==0:
                num_samples=250
            else:
                num_samples=0
            node_list,time_list= [],[]
            completed = 0
            
            log(f'------ {problem_type}, {sizes[i]} ------')
            for instance_count, instance in zip(range(num_samples), instances):
                _, _, _, _, _ = env.reset(instance)
                _, _, _, done, info = env.step({})
                
                node_list.append(info['nb_nodes'])
                time_list.append(info['time'])
                if info['time'] < 0.99*TIME_LIMIT:
                    completed += done

            log(f" nb nodes    {int(np.mean(node_list)): >4d} +- {int(np.std(node_list)): >4d} |  time {np.mean(time_list): >6.2f} | completed   {int(completed) : >2d} +- {np.std(time_list): >6.2f}")

    log('End of evaluation.')   
