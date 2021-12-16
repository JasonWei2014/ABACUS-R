from time import time
t000 = time()

import os
import sys
sys.path.append("../utils")
import jsonlines

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import Logger
from protein_utils import ENCODEAA2NUM

import numpy as np
import pandas as pd


protAlphabet = 'ACDEFGHIKLMNPQRSTVWY'
encodenum2AA = {x[1]: x[0] for x in ENCODEAA2NUM.items()}

def savemuts_inf(chain_seq, nat_logp, nat_entropy, nat_sing_logp, nat_logits, epoch_logits, epoch_logp, epoch_entropy, epoch_single_logp,
                 postmuts_aa, premuts_aa, available_muts, basename, outputroot):

    with jsonlines.open(os.path.join(outputroot, basename + "_multiscan_result.jsonl"), mode="w") as writer:
        nativedic = {"pdbname": basename,
                   "native_seq": "".join(list(map(lambda x: protAlphabet[x], chain_seq))),
                   "native_logp": nat_logp,
                   "native_entropy": nat_entropy,
                   "native_pos_logp": nat_sing_logp,
                   "native_logits": nat_logits
                   }
        writer.write(nativedic)

        output_order = np.argsort(epoch_logp)[::-1]

        for mutation_id in output_order:
            mutsdic = {"mut_sites": available_muts[mutation_id],
                       "premuts_aa": list(map(lambda x: protAlphabet[x], premuts_aa[mutation_id])),
                       "postmuts_aa": list(map(lambda x: protAlphabet[x], postmuts_aa[mutation_id])),
                       "postmuts_logp": epoch_logp[mutation_id],
                       "postmuts_entropy": epoch_entropy[mutation_id],
                       "postmuts_logits": epoch_logits[mutation_id],
                       "postmuts_single_logp": epoch_single_logp[mutation_id]
                       }
            writer.write(mutsdic)

    print("Output", basename, "scanning file")


def update_dateset(current_seq, chain_knn_inf, vocab_size=20, other_inf = None):
    current_seq_extra = torch.cat([current_seq, torch.randint_like(current_seq, vocab_size)[:1]], dim=0)
    # cat for -1
    chain_knn_inf["knnAAtype"] = torch.stack([current_seq_extra[knnpos] for knnpos in chain_knn_inf["knnpos"]], dim=0)
    if other_inf is not None and len(other_inf) != 0:
        other_idx = other_inf[:,:2].cpu().tolist()
        if torch.cuda.is_available():
            other_idx = torch.LongTensor(other_idx).cuda()
            other_aa = other_inf[:, 2].cuda()
        chain_knn_inf["knnAAtype"].index_put_(tuple(other_idx.t()), other_aa)

    return chain_knn_inf


def disturb_step(temperature, new_seq, seq_length):
    mut_index = np.random.choice(np.arange(0, seq_length), int(seq_length * 0.01 * temperature), replace=False)
    if len(mut_index) != 0:
        mut_index = torch.tensor(mut_index, device=new_seq.device)
        new_seq[mut_index] = torch.randint(0, 20, size=[mut_index.shape[0]], device=new_seq.device)
    return new_seq

def multiscan(config, model, checkpoint, dataset, device_ids):
    test_params = config["test_params"]

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model=model)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    model.eval()

    for batch_idx, (chain_seq, chain_cent_inf, chain_knn_inf) in enumerate(dataloader):

        seq_len = chain_seq.shape[1]
        print(seq_len)

        #### native inf
        if torch.cuda.is_available():

            nat_seq = chain_seq[0].cuda()

            nat_cent_inf = {
                'pdbname': chain_cent_inf['pdbname'],
                'node_dihedral': {key: value[0].cuda() for key, value in chain_cent_inf['node_dihedral'].items()},
                'dist': chain_cent_inf['dist'][0].cuda()
            }
            nat_knn_inf = {key: value[0].cuda() if isinstance(value, torch.Tensor) else value for key, value in chain_knn_inf.items()}

        ### if consider other chain inf
        if config["preprocess_params"]["otherchain"]:
            other_inf = chain_cent_inf["otherchain_idx"].squeeze(0)
            nat_knn_inf = update_dateset(nat_seq, nat_knn_inf, other_inf = other_inf)
        else:
            nat_knn_inf = update_dateset(nat_seq, nat_knn_inf)

        nat_output = model(nat_cent_inf, nat_knn_inf)

        nat_output = {
            'logits': nat_output[:, :20],
            'bfactor': nat_output[:, 20:21],
            'ss3': nat_output[:, 21:24],
            'ss8': nat_output[:, 24:32],
            'rsa': nat_output[:, 32:33],
            'k1k2': nat_output[:, 33:]
        }

        ## nat logp
        nat_logits = nat_output['logits']
        nat_logp = torch.log(F.softmax(nat_output['logits'], dim=1).gather(1, nat_seq.reshape(-1, 1)).t()[0]).mean().cpu().item()

        nat_entropy = torch.stack(
            [torch.distributions.Categorical(probs=F.softmax(nat_output['logits'], dim=1)[i]).entropy() for i in range(F.softmax(nat_output['logits'], dim=1).shape[0])]
            , dim=0).mean().cpu().item()
        nat_sing_logp = torch.log(F.softmax(nat_output['logits'], dim=1).gather(1, nat_seq.reshape(-1, 1)).t()[0]).cpu().detach().numpy().tolist()

        # nat_softmax = F.softmax(nat_output['logits'], dim=1).cpu().detach().numpy().tolist()
        nat_output = None

        if test_params["verbose"]:
            print("gt_logp: %06f; gt_entropy: %06f" % (nat_logp, nat_entropy))

        from itertools import combinations

        allsites = test_params["allmutsites"]
        if allsites == "ALL":
            allsites = np.arange(0, len(nat_cent_inf["pdbname"]), 1)
        else:
            allsites = [nat_cent_inf["pdbname"].index((nat_cent_inf["pdbname"][0][0].split("+")[0] + "+" + str(node),)) for node in allsites]
        print(allsites)
        mutsites_num = test_params["mutsites_num"]

        available_muts = list(combinations(allsites, mutsites_num))
        print(len(available_muts))

        epoch_logp = []
        epoch_entropy = []
        epoch_logits = []
        epoch_single_logp = []
        postmuts_aa = []
        premuts_aa = []
        mut_sites = []

        for mut_idx, multi_muts in enumerate(available_muts):
            term_muts = torch.LongTensor(multi_muts)
            tmp_seq = nat_seq.clone()
            for internal_updates in range(test_params["iter_num"]):

                if internal_updates == 0:
                    new_muts = torch.argmax(F.softmax(nat_logits, dim=1), 1)
                    tmp_seq[term_muts] = new_muts[term_muts]
                    new_seqs = tmp_seq
                else:
                    new_muts = torch.argmax(F.softmax(muts_output['logits'], dim=1), 1)
                    new_seqs[term_muts] = new_muts[term_muts]
                    
                muts_knn_inf = update_dateset(new_seqs, nat_knn_inf, other_inf = other_inf)
                muts_output = model(nat_cent_inf, muts_knn_inf)

                muts_output = {
                    'logits': muts_output[:, :20],
                    'bfactor': muts_output[:, 20:21],
                    'ss3': muts_output[:, 21:24],
                    'ss8': muts_output[:, 24:32],
                    'rsa': muts_output[:, 32:33],
                    'k1k2': muts_output[:, 33:]
                }

                muts_logits = muts_output['logits']
                muts_logp = torch.log(
                    F.softmax(muts_logits, dim=1).gather(1, new_seqs.reshape(-1, 1)).t()[0]).mean().cpu().item()

                muts_entropy = torch.stack(
                    [torch.distributions.Categorical(probs=F.softmax(muts_logits, dim=1)[i]).entropy() for i in
                     range(F.softmax(muts_logits, dim=1).shape[0])]
                    , dim=0).mean().cpu().item()

                muts_sing_logp = torch.log(
                    F.softmax(muts_logits, dim=1).gather(1, new_seqs.reshape(-1, 1)).t()[0]).cpu().detach().numpy().tolist()

                # muts_softmax = F.softmax(muts_logits, dim=1).cpu().detach().numpy().tolist()
                if test_params["verbose"]:
                    print("[epoch:%06d/iter:%02d] gt_logp: %06f; gt_entropy: %06f" % (mut_idx, internal_updates, muts_logp, muts_entropy))


            epoch_logits.append(muts_logits.cpu().detach().numpy().tolist())
            epoch_logp.append(muts_logp)
            epoch_entropy.append(muts_entropy)
            epoch_single_logp.append(muts_sing_logp)
            mut_sites.append(np.array(nat_cent_inf["pdbname"]).reshape(-1,)[list(multi_muts)].tolist())
            postmuts_aa.append(new_seqs[term_muts].cpu().detach().numpy().tolist())
            premuts_aa.append(nat_seq[term_muts].cpu().detach().numpy().tolist())

        savemuts_inf(nat_seq, nat_logp, nat_entropy, nat_sing_logp, nat_logits.cpu().detach().numpy().tolist(),
                     epoch_logits, epoch_logp, epoch_entropy, epoch_single_logp, postmuts_aa, premuts_aa, mut_sites,
                     outputroot= test_params["outputroot"], basename = nat_cent_inf["pdbname"][0][0].split("+")[0] + "_" + test_params["suffix"])
