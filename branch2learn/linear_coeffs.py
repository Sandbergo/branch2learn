import os
import torch
from pathlib import Path
import numpy as np

from models.mlp import MLP1Policy


if __name__ == '__main__':

    policy =  MLP1Policy()

    model_filename = f'branch2learn/models/mlp1/mlp1_cauctions.pkl'
    policy.load_state_dict(torch.load(model_filename))
    policy.eval()
    for param in policy.parameters():
        try:
            params = param.data[0].tolist()
            params = [params[0]] + params[5:]
            normalized = [round(2*(ele-min(params))/(max(params)-min(params))-1,2) for ele in params]
            print(params)
            print(normalized)
        except Exception as e:
            pass

    model_filename = f'branch2learn/models/mlp1/mlp1_setcover.pkl'
    policy.load_state_dict(torch.load(model_filename))
    policy.eval()
    for param in policy.parameters():
        try:
            params = param.data[0].tolist()
            params = [params[0]] + params[5:]
            normalized = [round(2*(ele-min(params))/(max(params)-min(params))-1,2) for ele in params]
            print(params)
            print(normalized)
        except Exception as e:
            pass