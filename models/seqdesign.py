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

def writeout_file(chain_seq, nat_logp, nat_entropy, nat_sing_logp, nat_softmax, nat_logits,
                  seq_list, similarity_old_new_list, similarity_new_nat_list, entropy_list, logp_list, logits_list,
                  epoch_seqlist, epoch_softmax, epoch_entropy, epoch_logp, epoch_logits, epoch_simi, epoch_sing_logp, mut_list,
                  basename, outputroot, save_all2json = True):

    with jsonlines.open(os.path.join(outputroot, basename + "_seq_design.jsonl"), mode="w") as writer:
        allseq_dic = {"pdbname": basename,
                   "native_seq": "".join(list(map(lambda x: protAlphabet[x], chain_seq))),
                   "native_logp":nat_logp,
                   "native_logits":nat_logits,
                   "native_entropy": nat_entropy,
                   "native_pos_logp":nat_sing_logp,
                   "native_softmax": nat_softmax,
                   "design_seq": epoch_seqlist,
                   "design_seq_logp": epoch_logp,
                   "design_seq_logits": epoch_logits,
                   "design_seq_entropy": epoch_entropy,
                   "design_simi": epoch_simi,
                   "design_pos_logp": epoch_sing_logp,
                   "design_softmax": epoch_softmax
                   }
        for term_name, term_value in allseq_dic.items():
            writer.write({term_name: term_value})

    print("Output", basename, "design file")

    if save_all2json == True:
        with jsonlines.open(os.path.join(outputroot, basename + "_all_inf.jsonl"), mode="w") as writer:
            alldic = {"pdbname": basename,
                       "all_seq": seq_list,
                       "identity_old_new": similarity_old_new_list,
                       "identity_new_nat": similarity_new_nat_list,
                       "entropy_list": entropy_list,
                       "logp_list": logp_list,
                       "logits_list": logits_list,
                       "confuse_site": mut_list
                       }
            for term_name, term_value in alldic.items():
                writer.write({term_name: term_value})

        print("Output", basename, "all_inf file")


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


def read_fasta_file(filename):
    seq_lists = []

    with open(filename) as sp_sf:
        spseq_flines = sp_sf.readlines()

    for line in spseq_flines:
        if not line.startswith(">"):
            seq_lists.append(line.strip())
        else:
            continue

    return seq_lists[0]


def seqdesign(config, model, checkpoint, dataset, device_ids):

    test_params = config["test_params"]

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model=model)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    model.eval()

    for batch_idx, (chain_seq, chain_cent_inf, chain_knn_inf) in enumerate(dataloader):

        seq_len = chain_seq.shape[1]
        print(chain_cent_inf['pdbname'][0][0].split("+")[0], seq_len)

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

        alpha = test_params["alpha"]
        ## nat logp
        nat_logits = nat_output['logits'].gather(1, nat_seq.reshape(-1, 1)).t()[0].sum().cpu().item()
        nat_logp = torch.log(F.softmax(nat_output['logits'], dim=1).gather(1, nat_seq.reshape(-1, 1)).t()[0]).sum().cpu().item()
        nat_entropy = torch.stack(
            [torch.distributions.Categorical(probs=F.softmax(nat_output['logits'], dim=1)[i]).entropy() for i in range(F.softmax(nat_output['logits'], dim=1).shape[0])]
            , dim=0).mean().cpu().item()
        nat_sing_logp = torch.log(F.softmax(nat_output['logits'], dim=1).gather(1, nat_seq.reshape(-1, 1)).t()[0]).cpu().detach().numpy().tolist()

        nat_softmax = F.softmax(nat_output['logits'], dim=1).cpu().detach().numpy().tolist()

        if torch.cuda.is_available():
            nat_seq = nat_seq.cpu()
            nat_cent_inf = {
                'pdbname': nat_cent_inf['pdbname'],
                'node_dihedral': {key: value[0].cpu() for key, value in nat_cent_inf['node_dihedral'].items()},
                'dist': nat_cent_inf['dist'][0].cpu()
            }
            nat_knn_inf = {key: value[0].cpu() if isinstance(value, torch.Tensor) else value for key, value in nat_knn_inf.items()}
            nat_output = None

        if test_params["verbose"]:
            print("gt_logp: %06f; gt_entropy: %06f; gt_logits: %06f" % (nat_logp, nat_entropy, nat_logits))


    #### design inf
        if torch.cuda.is_available():
            chain_seq = chain_seq[0].cuda()
            chain_cent_inf = {
                'pdbname': chain_cent_inf['pdbname'],
                'node_dihedral': {key: value[0].cuda() for key, value in chain_cent_inf['node_dihedral'].items()},
                'dist': chain_cent_inf['dist'][0].cuda()
            }

            chain_knn_inf = {key: value[0].cuda() if isinstance(value, torch.Tensor) else value for key, value in chain_knn_inf.items()}

        if test_params["random_init"]:
            current_seq = torch.randint_like(chain_seq, config["model_params"]["transformer_params"]["vocab_size"])
        elif test_params["struct_init"]:
            current_seq = nat_seq.cuda()
        elif test_params["sp_seqinit"] is not False:
            sp_seq_file = os.path.join(config["preprocess_params"]["pdbdataroot"],
                                   nat_cent_inf["pdbname"][0][0].split("+")[0] + "_spseqs")
            sp_init_seq = read_fasta_file(sp_seq_file)

            if torch.cuda.is_available():
                current_seq = torch.Tensor([ENCODEAA2NUM[aa] for aa in sp_init_seq]).long().cuda()
            else:
                current_seq = torch.Tensor([ENCODEAA2NUM[aa] for aa in sp_init_seq]).long()
        else:
            raise Exception("No initial sequence!")
        ### if specify aa type
        if test_params["sp_aa"]:
            sp_file = os.path.join(config["preprocess_params"]["pdbdataroot"], nat_cent_inf["pdbname"][0][0].split("+")[0] + "_spfile")

            with open(sp_file) as spf:
                spflines = spf.readlines()

            sp_list = [x.strip().split(",") for x in spflines]
            sp_list = np.array([[nat_cent_inf["pdbname"].index((nat_cent_inf["pdbname"][0][0][:-1] + spinf[0], )), spinf[1]] for spinf in sp_list])
            if torch.cuda.is_available():
                sp_index = torch.LongTensor(torch.from_numpy(sp_list[:, 0].astype(int))).cuda()
                sp_aa = torch.Tensor([ENCODEAA2NUM[aa] for aa in sp_list[:, 1]]).long().cuda()

            current_seq.index_put_((sp_index,), sp_aa)

        # if test_params["otherchain"]:
        if config["preprocess_params"]["otherchain"]:
            chain_knn_inf = update_dateset(current_seq, chain_knn_inf, other_inf = other_inf)
        else:
            chain_knn_inf = update_dateset(current_seq, chain_knn_inf)

        low = test_params["low"]
        high = test_params["high"]
        low_part = 100 - 100 * (seq_len - 1) / seq_len

        similarity_old_new_list = []
        similarity_new_nat_list = []
        logp_list = []
        logits_list = []
        entropy_list = []
        seq_list = []
        mut_list = []

        epoch_seqlist = []
        epoch_softmax = []
        epoch_entropy = []
        epoch_logp = []
        epoch_logits = []
        epoch_simi = []
        epoch_sing_logp = []

        for epoch in range(test_params["num_epochs"]):

            single_old_new_list = []
            single_new_nat_list = []
            single_logp_list = []
            single_logits_list = []
            single_entropy_list = []
            single_seq_list = []
            single_mut_list = []

            current_temp = high
            current_part = 100
            best_entropy = 1000
            best_logp = -10000
            best_simi = -100
            best_sing_logp = None
            best_probs = None
            best_seq = None
            similarity_old_new_cache =[]

            for iter in range(test_params["num_iters"]):

                output = model(chain_cent_inf, chain_knn_inf)

                output = {
                    'logits': output[:, :20],
                    'bfactor': output[:, 20:21],
                    'ss3': output[:, 21:24],
                    'ss8': output[:, 24:32],
                    'rsa': output[:, 32:33],
                    'k1k2': output[:, 33:]
                }

                ## softmax
                logits = output["logits"]
                probs = F.softmax(logits, dim=1)

                entropy = torch.stack([torch.distributions.Categorical(probs=probs[i]).entropy() for i in range(probs.shape[0])], dim=0)
                # ca
                all_entropy = entropy.mean().item()
                single_entropy_list.append(all_entropy)

                if test_params["max_sample"] == True:
                    logp = torch.log(probs.max(dim=1)[0]).sum().item()
                    seq_logits = logits.gather(1, current_seq.reshape(-1, 1)).t()[0].sum().cpu().item()
                    new_seq = torch.argmax(logits, dim=1)
                    single_logp_list.append(logp)
                    single_logits_list.append(seq_logits)

                else:
                    samp_probs = torch.where(probs > 0.001, probs, torch.zeros_like(probs))
                    samp_probs = torch.pow(samp_probs, alpha) / torch.pow(samp_probs, alpha).sum(1).unsqueeze(1)
                    aaidx = torch.distributions.Categorical(probs = samp_probs).sample()
                    logp = torch.log(torch.gather(probs, 1, aaidx.reshape(-1, 1))).mean().item()
                    seq_logits = logits.gather(1, aaidx.reshape(-1, 1)).t()[0].sum().cpu().item()
                    new_seq = aaidx
                    single_logp_list.append(logp)
                    single_logits_list.append(seq_logits)

                single_seq_list.append(new_seq.cpu().detach().numpy().tolist())

                similarity_old_new = (current_seq == new_seq).float().mean().item()
                diff_old_new_idx = (current_seq != new_seq)
                single_old_new_list.append(similarity_old_new)
                similarity_old_new_cache.append(similarity_old_new)
                similarity_new_nat = (chain_seq == new_seq).float().mean().item()
                single_new_nat_list.append(similarity_new_nat)

                if test_params["verbose"]:
                    print("[epoch:%02d/iter:%02d] old-new: %06f; new-gt: %06f; entropy: %06f; logP: %06f; logits: %06f" %
                          (epoch, iter, similarity_old_new, similarity_new_nat, all_entropy, logp, seq_logits))

                ### logp
                if test_params["savemaxlogp"] == True:
                    # if logp > best_logp:
                    if similarity_new_nat > best_simi:
                        best_simi = similarity_new_nat
                        best_seq = new_seq
                        best_entropy = all_entropy
                        best_logp = logp
                        best_logits = seq_logits
                        if test_params["max_sample"] == True:
                            best_sing_logp = torch.log(probs.max(dim=1)[0]).cpu().detach().numpy().tolist()
                        else:
                            best_sing_logp = torch.log(torch.gather(probs, 1, aaidx.reshape(-1, 1))).reshape(-1).cpu().detach().numpy().tolist()

                        best_probs = probs
                else:
                    best_simi = similarity_new_nat
                    best_seq = new_seq
                    best_entropy = all_entropy
                    best_logp = logp
                    best_logits = seq_logits
                    if test_params["max_sample"] == True:
                        best_sing_logp = torch.log(probs.max(dim=1)[0]).cpu().detach().numpy().tolist()
                    else:
                        best_sing_logp = torch.log(torch.gather(probs, 1, aaidx.reshape(-1, 1))).reshape(
                            -1).cpu().detach().numpy().tolist()

                    best_probs = probs

                #### update part and prepare

                if iter > (test_params["num_iters"] - test_params["low_keep"]):

                    if 1 - similarity_old_new < 1e-6:
                        keep_idx = np.delete(np.arange(0, seq_len), (iter - (test_params["num_iters"] - test_params["low_keep"]) - 1) % seq_len)
                    else:
                        mut_site = np.arange(0, seq_len)[diff_old_new_idx.cpu().detach().numpy()]
                        single_mut_list.extend(np.array(nat_cent_inf["pdbname"])[mut_site].reshape(1,-1)[0])
                        keep_idx = np.delete(np.arange(0, seq_len), np.random.choice(mut_site, 1))
                else:
                    keep_idx = np.random.choice(np.arange(0, seq_len), int(seq_len * 0.01 * (100 - current_part)), replace=False)

                keep_idx = torch.tensor(keep_idx, device=new_seq.device)

                new_seq[keep_idx] = current_seq[keep_idx]
                current_seq = new_seq

                current_seq = disturb_step(current_temp, current_seq, seq_len)
                if iter == 0 and high == 0:
                    current_seq = disturb_step(test_params["part_mut"], current_seq, seq_len)

                if test_params["sp_aa"]:
                    current_seq.index_put_([sp_index], sp_aa)

                if config["preprocess_params"]["otherchain"]:
                    chain_knn_inf = update_dateset(current_seq, chain_knn_inf, other_inf=other_inf)
                else:
                    chain_knn_inf = update_dateset(current_seq, chain_knn_inf)

                ## quench
                if current_temp > low:
                    if test_params["mutdecay"] == "linear":
                        current_temp = current_temp - (high - low) / (test_params["num_iters"] - test_params["low_iter"])
                    elif test_params["mutdecay"] == "exp":
                        current_temp = high * np.exp((np.log(high - low) / (test_params["num_iters"] - test_params["low_iter"])) * -iter)

                else:
                    current_temp = low

                if iter < (test_params["num_iters"] - test_params["low_iter"]):
                    current_part = test_params["part_mut"]
                elif iter > (test_params["num_iters"] - test_params["low_iter"]) and iter < (test_params["num_iters"] - test_params["low_keep"]):

                    if test_params["stepdecay"] == "linear":
                        current_part = current_part - (test_params["part_mut"] - low_part)/(test_params["low_iter"] - test_params["low_keep"])
                    elif test_params["stepdecay"] == "exp":
                        current_part = test_params["part_mut"] * np.exp((np.log(test_params["part_mut"] - low_part) / (test_params["low_iter"] - test_params["low_keep"])) *
                                                                         -(iter + test_params["low_iter"] - test_params["num_iters"]))

                elif iter > (test_params["num_iters"] - test_params["low_keep"]):
                    current_part = low_part


                if test_params["early_stop"]:
                    if torch.count_nonzero((1.0 - torch.Tensor(similarity_old_new_cache[-20:])) < 1e-5).item() > 19:
                        if test_params["verbose"]:
                            print("[epoch:%02d] lowest-entropy: %06f; highest-logP: %06f; highest-logits: %06f" %
                                  (epoch, best_entropy, best_logp, best_logits))
                        break

                if iter == test_params["num_iters"] - 1:
                    if test_params["verbose"]:
                        print("[epoch:%02d] lowest-entropy: %06f; highest-logP: %06f; highest-logits: %06f" %
                              (epoch, best_entropy, best_logp, best_logits))

            epoch_seqlist.append("".join(list(map(lambda x: protAlphabet[x], best_seq.cpu().numpy().tolist()))))
            epoch_softmax.append(best_probs.cpu().detach().numpy().tolist())
            epoch_entropy.append(best_entropy)
            epoch_logp.append(best_logp)
            epoch_logits.append(best_logits)
            epoch_simi.append(best_simi)
            epoch_sing_logp.append(best_sing_logp)
            similarity_old_new_list.append(single_old_new_list)
            similarity_new_nat_list.append(single_new_nat_list)
            logp_list.append(single_logp_list)
            logits_list.append(single_logits_list)
            entropy_list.append(single_entropy_list)
            seq_list.append(single_seq_list)
            mut_list.append(single_mut_list)

        writeout_file(chain_seq, nat_logp, nat_entropy, nat_sing_logp, nat_softmax, nat_logits,
                      seq_list, similarity_old_new_list, similarity_new_nat_list, entropy_list, logp_list, logits_list,
                      epoch_seqlist, epoch_softmax, epoch_entropy, epoch_logp, epoch_logits, epoch_simi, epoch_sing_logp, mut_list,
                      outputroot= test_params["outputroot"], basename = nat_cent_inf["pdbname"][0][0].split("+")[0] + "_" + test_params["suffix"],
                      save_all2json=test_params["all_inf"])

