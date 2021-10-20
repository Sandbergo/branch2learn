"""
Policy evaluation script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import argparse
from pathlib import Path

import numpy as np
import ecole

from utilities.general import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--brancher',
        help='Brancher name.',
        choices=['pc', 'fsb', 'rpc'],
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

    POLICY_DICT = {'fsb': 'fullstrong',
                   'pc': 'pscost',
                   'rpc': 'relpscost'}
    TIME_LIMIT = 2700

    Path('branch2learn/log/').mkdir(exist_ok=True)
    log = Logger(filename='branch2learn/log/05_evaluate_standard')

    log(f'Policy:   {args.brancher}')
    policy = POLICY_DICT[args.brancher]

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 
                       'limits/time': TIME_LIMIT, f'branching/{policy}/priority': 666666}

    env = ecole.environment.Configuring(
        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                              "time": ecole.reward.SolvingTime().cumsum()},
        scip_params=scip_parameters)

    problems = [str(path) for path in Path(
        f'branch2learn/data/instances/{args.problem}/eval/').glob('instance_*.lp')]

    node_list = []
    time_list = []
    completed = 0

    for instance_count, instance in zip(range(num_samples), instances):
        _, _, _, _, _ = env.reset(instance)
        _, _, _, done, info = env.step({})
        node_list.append(info['nb_nodes'])
        time_list.append(info['time'])
        if info['time'] < 0.99*TIME_LIMIT:
            completed += done

    log(f"  time {np.mean(time_list): >6.2f} \pm {np.std(time_list): >6.2f}  "
        f"| nb nodes    {int(np.mean(node_list)): >4d} \pm {int(np.std(node_list)): >4d} "
        f"| time/node  {1000*np.mean(time_list)/np.mean(node_list): >6.1f} ms "
        f"| completed   {completed : >2d}")

    log('End of evaluation.')   
