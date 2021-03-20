import numpy as np
import torch
import torch_geometric
import os
import argparse
from pathlib import Path

from utilities_general import Logger
from utilities_model import process
from utilities_data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy


def process(model, dataloader, top_k):
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
    for batch in dataloader:
        cand_features, n_cands, best_cands, cand_scores, weights  = map(lambda x:x.to(device), batch)
        batched_states = (cand_features)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            logits = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now

        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == '__main__':
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

    EARLY_STOPPING = 20
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
    log = Logger(filename='branch2learn/log/02_train')

    log(f'Model:   {args.model}')
    log(f'Problem: {PROBLEM}')
    log(f'Device:  {DEVICE}')
    log(str(policy))


    # --- TEST --- #
    sample_files = [str(path) for path in Path(
        f'branch2learn/data/samples/{PROBLEM}/test'
        ).glob('sample_*.pkl')]
    test_files = sample_files[:int(0.8*len(sample_files))]

    test_data = GraphDataset(train_files)
    test_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=False)

    model_filename = f'branch2learn/models/{args.model}/{args.model}_{PROBLEM}.pkl'
    policy.restore_state(model_filename)

    log('Beginning testing')
    test_kacc = process(policy=policy, data_loader=test_loader, device=DEVICE, optimizer=None)

    log(f'Test kacc: {test_kacc}')

    log('End of testing.')
