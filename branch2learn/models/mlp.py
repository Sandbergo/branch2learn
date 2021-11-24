"""
MLP models.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import torch


class MLP1Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        var_nfeats = 19

        self.output_module = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        output = self.output_module(variable_features).squeeze(-1)
        return output


class MLP2Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        var_nfeats = 19

        self.output_module = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        output = self.output_module(variable_features).squeeze(-1)
        return output


class MLP3Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        var_nfeats = 19

        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        output = self.output_module(self.var_embedding(variable_features)).squeeze(-1)
        return output
