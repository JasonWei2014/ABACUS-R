from time import time
t000 = time()
import os
import sys
sys.path.append("../utils")
import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm import trange
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from logger import Logger

import numpy as np
from scipy import stats
from sklearn.metrics import precision_score
import pandas as pd

from protein_utils import ENCODEAA2NUM


def validate(config, model, checkpoint, dataset, device_ids):
    validate_params = config["validate_params"]

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model=model)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=6, drop_last=True)

    model = model.eval()

    AAtype = []
    ss3 = []
    ss8 = []
    bfactor = []
    rsa = []
    k1 = []
    k2 = []
    bin_k1 = []
    bin_k2 = []


    for batch_idx, (label, cent_inf, knn_inf) in enumerate(dataloader):
        if batch_idx * 512 >= validate_params["num_samples"]:
            break

        if torch.cuda.is_available():
            label = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in label.items()}
            cent_inf = {
                'pdbname': cent_inf['pdbname'],
                'node_dihedral': {key: value.cuda() for key, value in cent_inf['node_dihedral'].items()},
                'dist': cent_inf['dist'].cuda()
            }
            knn_inf = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in knn_inf.items()}

        output = model(cent_inf, knn_inf)

        output = {
            'logits': output[:, :20],
            'bfactor': output[:, 20:21],
            'ss3': output[:, 21:24],
            'ss8': output[:, 24:32],
            'rsa': output[:, 32:33],
            'k1k2': output[:, 33:]
        }

        pred = torch.argmax(output['logits'], dim=1)
        print(torch.nn.functional.cross_entropy(output['logits'], label['centralAA']))
        pred_ss3 = torch.argmax(output['ss3'], dim=1)
        pred_ss8 = torch.argmax(output['ss8'], dim=1)
        k1k2_delta = (F.tanh(output['k1k2']).abs() - label['k1k2'].abs()) * label['k1k2_mask']
        k1k2_add = (F.tanh(output['k1k2']).abs() + label['k1k2'].abs()) * label['k1k2_mask']

        AAtype.append((pred == label["centralAA"]).float().mean().item())
        ss3.append((pred_ss3 == label["ss3"]).float().mean().item())
        ss8.append((pred_ss8 == label["ss8"]).float().mean().item())
        bfactor.append((output["bfactor"].squeeze() - label["nodebfactor"]).abs().mean().item())
        rsa.append((output["rsa"].squeeze() - label["rsa"]).abs().mean().item())

        ## similar symbol
        homosymbol = (((F.tanh(output['k1k2']) > 0) & (label['k1k2'] > 0)) | ((F.tanh(output['k1k2']) < 0) & (label['k1k2'] < 0))) * label['k1k2_mask']
        ## diff symbol & diff < 180
        lowerheter = (homosymbol == False) & (((F.tanh(output['k1k2']).abs() + label['k1k2'].abs()) < 1.)) * label['k1k2_mask']
        ## diff symbol & diff > 180
        greaterheter = (homosymbol == False) & (((F.tanh(output['k1k2']).abs() + label['k1k2'].abs()) > 1.)) * label['k1k2_mask']
        truek1k2_delta = homosymbol * k1k2_delta + lowerheter * k1k2_add + greaterheter * (2. - k1k2_add)
        k1.append((truek1k2_delta[:, 0].abs().sum() / label['k1k2_mask'].sum()).item())
        k2.append((truek1k2_delta[:, 1].abs().sum() / label['k1k2_mask'].sum()).item())

        torsion_bin = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65]) / 2
        bin_k1.append([((((truek1k2_delta[:, 0] * 180) < each_bin) & ((truek1k2_delta[:, 0] * 180) > -each_bin) & label['k1k2_mask'][:, 0] & (pred == label["centralAA"])).sum() / (label['k1k2_mask'][:, 0] & (pred == label["centralAA"])).sum()).item() for each_bin in torsion_bin])
        bin_k2.append([((((truek1k2_delta[:, 1] * 180) < each_bin) & ((truek1k2_delta[:, 1] * 180) > -each_bin) & label['k1k2_mask'][:, 0] & (pred == label["centralAA"])).sum() / (label['k1k2_mask'][:, 1] & (pred == label["centralAA"])).sum()).item() for each_bin in torsion_bin])


    print("AAtype: %s" % np.array(AAtype).mean())
    print("ss3: %s" % np.array(ss3).mean())
    print("ss8: %s" % np.array(ss8).mean())
    print("bfactor: %s" % np.array(bfactor).mean())
    print("rsa: %s" % np.array(rsa).mean())
    print("k1: %s" % np.array(k1).mean())
    print("k2: %s" % np.array(k2).mean())
    print("K1 bin",np.array(bin_k1).mean(0))
    print("K2 bin",np.array(bin_k2).mean(0))
