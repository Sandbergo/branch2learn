"""
Model evalutation script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import os
import argparse
import time
from pathlib import Path

import scipy.stats as st
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric
import ecole

from utilities.general import Logger
from utilities.model import process
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Model name.",
        choices=["gnn1", "gnn2", "mlp1", "mlp2", "mlp3", "mlp4"],
    )
    parser.add_argument(
        "-p",
        "--problem",
        help="MILP instance type to process.",
        choices=["setcover", "cauctions", "facilities", "indset"],
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="CUDA GPU id (-1 for CPU).",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Random generator seed.",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    POLICY_DICT = {
        "mlp1": MLP1Policy(),
        "mlp2": MLP2Policy(),
        "mlp3": MLP3Policy(),
        "gnn1": GNN1Policy(),
        "gnn2": GNN2Policy(),
    }
    PROBLEM = args.problem
    TIME_LIMIT = 2700

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        DEVICE = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = POLICY_DICT[args.model].to(DEVICE)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    Path("branch2learn/log/").mkdir(exist_ok=True)
    log = Logger(filename="branch2learn/log/04_evaluate")

    log(f"Model:   {args.model}")
    log(f"Problem: {PROBLEM}")
    log(f"Device:  {DEVICE}")

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": TIME_LIMIT,
    }
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
        scip_params=scip_parameters,
    )

    problems = [
        str(path)
        for path in Path(f"branch2learn/data/instances/{args.problem}/eval/").glob(
            "instance_*.lp"
        )
    ]

    model_filename = f"branch2learn/models/{args.model}/{args.model}_{args.problem}.pkl"
    policy.load_state_dict(
        torch.load(model_filename, map_location=torch.device(DEVICE))
    )
    policy.eval()

    node_list = []
    time_list = []
    completed = 0

    for instance_count, instance in zip(range(num_samples), problems):
        observation, action_set, _, done, info = env.reset(instance)
        while not done:

            with torch.no_grad():
                observation = (
                    torch.as_tensor(
                        observation.row_features, dtype=torch.float32, device=DEVICE
                    ),
                    torch.as_tensor(
                        observation.edge_features.indices.astype(np.int64),
                        dtype=torch.int64,
                        device=DEVICE,
                    ),
                    torch.as_tensor(
                        observation.edge_features.values,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).view(-1, 1),
                    torch.as_tensor(
                        observation.column_features, dtype=torch.float32, device=DEVICE
                    ),
                )

                logits = policy(*observation)
                action = action_set[logits[action_set.astype(np.int64)].argmax()]
                observation, action_set, _, done, info = env.step(action)

        node_list.append(info["nb_nodes"])
        time_list.append(info["time"])
        if info["time"] < 0.99 * TIME_LIMIT:
            completed += done

    mean_time = np.mean(time_list)
    mean_node = np.mean(node_list)
    log(
        f"  time {mean_time: >6.2f} \pm {np.std(time_list): >6.2f}  "
        f"| nb nodes    {int(mean_node): >4d} \pm {int(np.std(node_list)): >4d} "
        f"| time/node  {1000*mean_time/mean_node: >6.1f} ms"
        f"| completed   {completed : >2d}"
    )

    log("End of evaluation.")


if __name__ == "__main__":
    main()
