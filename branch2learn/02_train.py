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

    LEARNING_RATE = 0.001
    NB_EPOCHS = 2
    PATIENCE = 10
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

    Path('examples/log/').mkdir(exist_ok=True)
    log = Logger(filename='examples/log/02_train')
    
    log(f'Model:   {args.model}')
    log(f'Problem: {PROBLEM}')
    log(f'Device:  {DEVICE}')
    log(f'Lr:      {LEARNING_RATE}')
    log(f'Epochs:  {NB_EPOCHS}')
    log(str(policy))


    # --- TRAIN --- #
    sample_files = [str(path) for path in Path(
        f'examples/data/samples/{PROBLEM}/train'
        ).glob('sample_*.pkl')]
    train_files = sample_files[:int(0.8*len(sample_files))]
    valid_files = sample_files[int(0.8*len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

    model_filename = f'examples/models/{args.model}/{args.model}_{PROBLEM}.pkl'
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=PATIENCE, verbose=True)

    best_loss = np.inf

    log('Beginning training')
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

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            policy.save_state(model_filename)
            log(f'  best model so far')
        else:
            plateau_count += 1
            if plateau_count % EARLY_STOPPING == 0:
                log(f'  {plateau_count} epochs without improvement, early stopping')
                break
            if plateau_count % PATIENCE == 0:
                LEARNING_RATE *= 0.2
                log(f'  {plateau_count} epochs without improvement, decrease lr to {LEARNING_RATE}')

        scheduler.step(valid_loss)

    policy.restore_state(model_filename)
    valid_loss, valid_acc = process(
        policy=policy, data_loader=valid_data,  device=DEVICE, optimizer=None)
    log(f'Best valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}')

    log(f'Saving model as {model_filename}')
    torch.save(policy.state_dict(), model_filename)
    log('End of training.')
