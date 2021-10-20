"""
Model training script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch_geometric

from utilities.general import Logger
from utilities.model import process
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy, GNNSPolicy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        help='Model name.',
        choices=['gnn1', 'gnn2', 'gnns', 'mlp1', 'mlp2', 'mlp3'],
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
        default=0,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    LEARNING_RATE = 0.001
    NB_EPOCHS = 100
    PATIENCE = 8
    EARLY_STOPPING = 16
    POLICY_DICT = {'mlp1': MLP1Policy(), 'mlp2': MLP2Policy(), 'mlp3': MLP3Policy(),
                   'gnn1': GNN1Policy(), 'gnn2': GNN2Policy()}
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
    log(f'Lr:      {LEARNING_RATE}')
    log(f'Epochs:  {NB_EPOCHS}')

    # --- TRAIN --- #
    train_files = [str(path) for path in Path(
        f'branch2learn/data/samples/{PROBLEM}/train'
        ).glob('sample_*.pkl')]
    valid_files = [str(path) for path in Path(
        f'branch2learn/data/samples/{PROBLEM}/valid'
        ).glob('sample_*.pkl')]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=32, shuffle=False)

    model_filename = f'branch2learn/models/{args.model}/{args.model}_{PROBLEM}.pkl'
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=PATIENCE, verbose=True)

    best_loss = np.inf

    log('Beginning training')

    train_loss, train_acc = process(
        policy=policy, data_loader=train_loader, device=DEVICE, optimizer=None)
    log(f'Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}')

    valid_loss, valid_acc = process(
        policy=policy, data_loader=valid_loader, device=DEVICE, optimizer=None)
    log(f'Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}')

    for epoch in range(NB_EPOCHS):
        log(f'Epoch {epoch+1}')

        train_loss, train_acc = process(
            policy=policy, data_loader=train_loader, device=DEVICE, optimizer=optimizer)
        log(f'Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}')

        valid_loss, valid_acc = process(
            policy=policy, data_loader=valid_loader, device=DEVICE, optimizer=None)
        log(f'Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}')

        if valid_loss < best_loss*0.999:
            plateau_count = 0
            best_loss = valid_loss
            torch.save(policy.state_dict(), model_filename)
            log('  best model so far')
        else:
            plateau_count += 1
            if plateau_count % EARLY_STOPPING == 0:
                log(f'  {plateau_count} epochs without improvement, early stopping')
                break
            if plateau_count % PATIENCE == 0:
                LEARNING_RATE *= 0.2
                log(f'  {plateau_count} epochs without improvement, decrease lr to {LEARNING_RATE}')

        scheduler.step(valid_loss)
    policy.load_state_dict(torch.load(model_filename))
    valid_loss, valid_acc = process(
        policy=policy, data_loader=valid_loader, device=DEVICE, optimizer=None)
    log(f'Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}')

    log(f'Saving model as {model_filename}')
    torch.save(policy.state_dict(), model_filename)

    log('End of training.\n\n')
