"""
Model training script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import os
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from utilities.general import Logger
from utilities.model import pad_tensor
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy


def process_kacc(
    policy, data_loader, device: str, top_k: List[int], optimizer=None
) -> List[float]:
    """
    Executes only a forward pass of model over the dataset and computes accuracy
    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    Return
    ------
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """

    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations)
            # according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            # Index the results by the candidates, and split and pad them

            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
            true_scores = true_scores.cpu().numpy()
            true_bestscore = true_bestscore.cpu().numpy()

            kacc = []
            for k in top_k:
                pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
                pred_top_k_true_scores = np.take_along_axis(
                    true_scores, pred_top_k, axis=1
                )
                kacc.append(
                    np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1))
                )
            kacc = np.asarray(kacc)

            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Model name.",
        choices=["gnn1", "gnn2", "mlp1", "mlp2", "mlp3"],
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
    log = Logger(filename="branch2learn/log/03_test")

    log(f"Model:   {args.model}")
    log(f"Problem: {PROBLEM}")
    log(f"Device:  {DEVICE}")

    # --- TEST --- #
    test_files = [
        str(path)
        for path in Path(f"branch2learn/data/samples/{PROBLEM}/test").glob(
            "sample_*.pkl"
        )
    ]

    test_data = GraphDataset(test_files)
    test_loader = torch_geometric.data.DataLoader(
        test_data, batch_size=32, shuffle=False
    )

    model_filename = f"branch2learn/models/{args.model}/{args.model}_{PROBLEM}.pkl"
    policy.load_state_dict(torch.load(model_filename))
    policy.eval()

    log("Beginning testing")
    test_kacc = process_kacc(
        policy=policy, data_loader=test_loader, device=DEVICE, top_k=range(1, 11)
    )
    test_kacc_round = 100 * np.round(test_kacc, 3)
    log(f"Test kacc: {test_kacc_round} %")
    log(
        f"{test_kacc_round[0]: >2.1f} % & {test_kacc_round[4]: >2.1f} %"
        f" & {test_kacc_round[9]: >2.1f} %"
    )

    log("End of testing.")
