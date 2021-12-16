import os
import time
import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from mingpt import GPT


class LocalTransformer(nn.Module):
    def __init__(self, vocab_size, n_output, n_seq=128, n_spatial=7, n_dihedral=15, n_layer=12, n_head=8, n_embd=256):
        super(LocalTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.n_output = n_output
        self.n_seq = n_seq

        self.token_emb = nn.Embedding(vocab_size + 1, n_embd)
        self.spatial_emb = nn.Linear(n_spatial, n_embd)
        self.dihedral_emb = nn.Linear(n_dihedral, n_embd)
        self.sequence_emb = nn.Embedding(n_seq + 1, n_embd)

        self.transformer = GPT(n_output=n_output, n_layer=n_layer, n_head=n_head, n_embd=n_embd * 4)

    def forward(self, cent_inf, knn_inf):
        batch_size = knn_inf['knnAAtype'].shape[0]
        n_node = knn_inf['knnAAtype'].shape[1]

        node_dihedral = cent_inf['node_dihedral']
        dihedral = torch.cat([node_dihedral['phi'], node_dihedral['psi'], node_dihedral['omega']], dim=1)
        knn_rep = sorted(list(set(knn_inf.keys()).difference(set(["knnpos", "knnind", "knnAAtype", "k_nndist"]))))


        # center node
        node_emb = self.token_emb(torch.tensor([self.vocab_size] * batch_size).cuda())
        if "dist" in cent_inf:
            spatial = torch.cat([cent_inf["dist"]] +
                                [torch.zeros_like(knn_inf[key])[:, 0:1] for key in knn_rep], dim=2)
        else:
            spatial = torch.cat([torch.zeros_like(knn_inf["dist"])[:, 0:1]] +
                                [torch.zeros_like(knn_inf[key])[:, 0:1] for key in knn_rep], dim=2)
        spatial_emb = self.spatial_emb(spatial[:, 0])
        dihedral_emb = self.dihedral_emb(dihedral)
        seq_emb = self.sequence_emb(torch.zeros_like(knn_inf["knnind"][:, 0] + self.n_seq // 2))
        cent_emb = torch.cat([node_emb, spatial_emb, dihedral_emb, seq_emb], dim=1)

        # knn node
        node_emb = self.token_emb(knn_inf['knnAAtype'])
        spatial = torch.cat([knn_inf['k_nndist']] +
                            [knn_inf[key] for key in knn_rep], dim=2)
        spatial_emb = self.spatial_emb(spatial)
        dihedral_emb = self.dihedral_emb(torch.zeros_like(dihedral).unsqueeze(dim=1).repeat(1, n_node, 1))
        seq_emb = self.sequence_emb(knn_inf["knnind"] + self.n_seq // 2)
        knn_emb = torch.cat([node_emb, spatial_emb, dihedral_emb, seq_emb], dim=2)

        embeddings = torch.cat([cent_emb.unsqueeze(dim=1), knn_emb], dim=1)

        # Transformer
        output = self.transformer(embeddings)[:, 0]

        return output