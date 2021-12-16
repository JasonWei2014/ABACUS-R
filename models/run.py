import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.append("../dataset")
import yaml
import torch
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from k1k2node_dataset import NodeDataset
from chain_dataset import ChainDataset
from local_integration import LocalTransformer
from train import train
from validate import validate
from seqdesign import seqdesign
from multiscan import multiscan


def setup_seed(seed=98):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed()

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "validate", "seqdesign", "multiscan"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    transformer = LocalTransformer(**config['model_params']['transformer_params'])

    if torch.cuda.is_available():
        transformer.to(opt.device_ids[0])
    if opt.verbose:
        print(transformer)

    if opt.mode == 'train' or opt.mode == 'validate':
        dataset = NodeDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    else:
        dataset = ChainDataset(**config['dataset_params'], dirroot=config['preprocess_params']['outputroot'], protein_list=config['preprocess_params']['filename'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, transformer, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'validate':
        print("Validating...")
        validate(config, transformer, opt.checkpoint, dataset, opt.device_ids)
    elif opt.mode == 'seqdesign':
        print("seqdesign...")
        if os.path.isdir(config['test_params']['outputroot']):
            pass
        else:
            os.mkdir(config['test_params']['outputroot'])
        seqdesign(config, transformer, opt.checkpoint, dataset, opt.device_ids)

    elif opt.mode == "multiscan":
        print("multiscan...")
        if os.path.isdir(config['test_params']['outputroot']):
            pass
        else:
            os.mkdir(config['test_params']['outputroot'])
        multiscan(config, transformer, opt.checkpoint, dataset, opt.device_ids)
