import os
import sys
sys.path.append("../utils")
import json

import pandas as pd
import numpy as np

from torch.utils import data
import torch

from protein_utils import ENCODEAA2NUM, PROTEINLETTER3TO1, RESIDUEMAXACC


def seq2ind(seq):
    protAlphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    return np.array([protAlphabet.find(n) for n in seq], dtype=np.int64)


def knnind(chainlist, pdbname, range=(-64, 63)):
    mychain = pdbname.split("_")[1].split("+")[0]
    centind = int(pdbname.split("_")[1].split("+")[1])
    knnind = np.array(
        list(map(lambda x: int(x[1]) - centind
        if x[0] == mychain and int(x[1]) - centind >= range[0] and int(x[1]) - centind <= range[1]
        else range[1] + 1, chainlist)), dtype=np.int64)
    return knnind


class NodeDataset(data.Dataset):
    def __init__(self, root_dir, basename, neighborhood_size, is_train=False, protein_list=None,
                 use_normalize=False, dist_mode='classic', use_trigonometric=False, old_features=True,
                 use_relative_rsa=False, pred_k1k2=False):
        self.root_dir = root_dir
        self.basename = basename
        self.neighborhood_size = neighborhood_size
        self.use_normalize = use_normalize
        self.dist_mode = dist_mode
        self.old_features = old_features
        self.use_trigonometric = use_trigonometric
        self.use_relative_rsa = use_relative_rsa
        self.pred_k1k2 = pred_k1k2
        self.all_data = []

        self.dist_mean = 7.77
        self.rsa_mean_std = [44.290062, 45.782705]  # calculated from trainset only
        self.bfactor_mean_std = [114.900829, 74.134204]  # calculated from trainset only

        self.protein_letters_1to3 = {x[1]: x[0] for x in PROTEINLETTER3TO1.items()}
        self.encodenum2AA = {x[1]: x[0] for x in ENCODEAA2NUM.items()}

        if protein_list is None:
            protein_list = "/home/liuyf/alldata/trainset.txt" if is_train else "/home/liuyf/alldata/test.txt"

        with open(protein_list, "r") as f:
            protein_list = f.readlines()
            protein_list = [x.strip() for x in protein_list]

        for protein in protein_list:
            path = os.path.join(root_dir, basename, protein, "pdbname.txt")
            k1k2 = os.path.join("/home/liuyf/alldata/pdbname_k1k2", protein, "k1k2.jsonl")
            if self.pred_k1k2:
                if os.path.isfile(k1k2):
                    pass
                else:
                    continue

            with open(path, "r") as f:
                pdbname_list = f.readlines()
                pdbname_list = [x.strip() for x in pdbname_list]
            self.all_data += pdbname_list

        self.protein_list_len = len(self.all_data)
        print(self.protein_list_len)

    def __getitem__(self, index):
        pdbname = self.all_data[index]
        protein = pdbname.split('+')[0]

        node_file = os.path.join(self.root_dir, self.basename, protein, "pdbname.txt")
        director_file = os.path.join(self.root_dir, self.basename, protein, "director.jsonl")
        angle_file = os.path.join(self.root_dir, self.basename, protein, "angle.jsonl")
        AAtype_file = os.path.join(self.root_dir, self.basename, protein, "AAtype.jsonl")
        dist_file = os.path.join(self.root_dir, self.basename, protein, "dist.jsonl")
        sphere3d_file = os.path.join(self.root_dir, self.basename, protein, "sphere3d.jsonl")
        allinternal_file = os.path.join(self.root_dir, self.basename, protein, "split_all_internal.jsonl")
        k1k2_file = os.path.join(self.root_dir, self.basename, protein, "k1k2.jsonl")

        with open(node_file) as nodf:
            nodflines = nodf.readlines()
        with open(director_file) as dirf:
            dirflines = dirf.readlines()

        if self.pred_k1k2:
            with open(k1k2_file) as k1k2f:
                klines = k1k2f.readlines()

        if self.old_features:
            with open(angle_file) as anf:
                anflines = anf.readlines()
            with open(AAtype_file) as aaf:
                aaflines = aaf.readlines()
            with open(dist_file) as disf:
                disflines = disf.readlines()
            with open(sphere3d_file) as sphf:
                sphflines = sphf.readlines()
        else:
            with open(allinternal_file) as itf:
                itflines = itf.readlines()

        pdbname_list = [x.strip() for x in nodflines]
        node = pdbname_list.index(pdbname)
        entry = json.loads(dirflines[node])
        if self.pred_k1k2:
            k1k2_entry = json.loads(klines[node])
        if self.old_features:
            angle_entry = json.loads(anflines[node])
            aa_entry = json.loads(aaflines[node])
            dist_entry = json.loads(disflines[node])
            sphere_entry = json.loads(sphflines[node])
        else:
            internal_entry = json.loads(itflines[node])

        # label
        label = {
            "centralAA": entry['centralAA'],
            "ss8": entry['ss8'],
            "ss3": entry['ss3'],
            "rsa": entry["rsa"],
            "nodebfactor": entry["nodebfactor"]
        }

        if self.use_normalize:
            if self.use_relative_rsa:
                label["rsa"] = label['rsa'] / RESIDUEMAXACC["Wilke"][self.protein_letters_1to3[self.encodenum2AA[entry["centralAA"]]]]
            else:
                label["rsa"] = (label['rsa'] - self.rsa_mean_std[0]) / self.rsa_mean_std[1]
            label["nodebfactor"] = (label["nodebfactor"] - self.bfactor_mean_std[0]) / self.bfactor_mean_std[1]

        if self.pred_k1k2:
            label["k1k2"] = np.array(k1k2_entry["k_nndist"], dtype=np.float32) / 180.0
            label["k1k2_mask"] = (label["k1k2"] >= -1.0) & (label["k1k2"] <= 1.0)

        node_diherdral = {
            'phi': np.array(entry["node_dihedral"]['phi'], dtype=np.float32),
            'psi': np.array(entry["node_dihedral"]['psi'], dtype=np.float32),
            'omega': np.array(entry["node_dihedral"]['omega'], dtype=np.float32)
        }

        if self.use_trigonometric:
            node_diherdral['phi'] = self._torsion_triangle(node_diherdral['phi']).reshape(-1)
            node_diherdral['psi'] = self._torsion_triangle(node_diherdral['psi']).reshape(-1)
            node_diherdral['omega'] = self._torsion_triangle(node_diherdral['omega']).reshape(-1)
        else:
            node_diherdral['phi'] = node_diherdral['phi'] / 180.0
            node_diherdral['psi'] = node_diherdral['psi'] / 180.0
            node_diherdral['omega'] = node_diherdral['omega'] / 180.0

        # cent_inf
        cent_inf = {
            "pdbname": entry["pdbname"],
            "node_dihedral": node_diherdral
        }

        if self.dist_mode == 'rbf':
            cent_inf["dist"] = self._rbf(np.array([0.0])).astype(np.float32)
        elif self.dist_mode == 'exp':
            cent_inf["dist"] = np.array([1.0], dtype=np.float32)[:, np.newaxis]
        elif self.dist_mode == 'classic':
            cent_inf["dist"] = np.array([1.0], dtype=np.float32)[:, np.newaxis]

        # knn inf
        if self.old_features:
            knn_inf = {
                "knnind": knnind(entry["k_nnidx"], entry["pdbname"])[:self.neighborhood_size],
                "knnAAtype": np.array(aa_entry["knnAAtype"])[:self.neighborhood_size],
                "k_nndist": np.array(dist_entry["k_nndist"], dtype=np.float32)[:self.neighborhood_size],
                "k_nnangle": np.array(angle_entry["k_nnangle"], dtype=np.float32)[:self.neighborhood_size],
                "k_nnsphere3d": np.array(sphere_entry["k_nnsphere3d"], dtype=np.float32)[:self.neighborhood_size]
            }

            if self.dist_mode == 'rbf':
                knn_inf["k_nndist"] = self._rbf(knn_inf["k_nndist"]).astype(np.float32)
            elif self.dist_mode == 'exp':
                knn_inf["k_nndist"] = np.exp(- knn_inf["k_nndist"] / self.dist_mean)[:, np.newaxis]
            elif self.dist_mode == 'classic':
                knn_inf["k_nndist"] = knn_inf["k_nndist"][:, np.newaxis]  # modify at 2021/07/26

            if self.use_trigonometric:
                knn_inf["k_nnangle"] = self._angle_triangle(knn_inf["k_nnangle"])
                knn_inf["k_nnsphere3d"] = self._angle_triangle(knn_inf["k_nnsphere3d"])


        else:
            knn_inf = {
                "knnind": knnind(internal_entry["k_nnid"], entry["pdbname"])[:self.neighborhood_size],
                "knnAAtype": np.array(internal_entry["k_nnAA"])[:self.neighborhood_size],
                "k_nndist": np.array(internal_entry["k_nndist"], dtype=np.float32)[:self.neighborhood_size],
                "k_nnomega": np.array(internal_entry["k_nnomega"], dtype=np.float32)[:self.neighborhood_size],
                "k_nntheta1": np.array(internal_entry["k_nntheta1"], dtype=np.float32)[:self.neighborhood_size],
                "k_nntheta2": np.array(internal_entry["k_nntheta2"], dtype=np.float32)[:self.neighborhood_size],
                "k_nndelta1": np.array(internal_entry["k_nndelta1"], dtype=np.float32)[:self.neighborhood_size],
                "k_nndelta2": np.array(internal_entry["k_nndelta2"], dtype=np.float32)[:self.neighborhood_size],
                "k_nnphi1": np.array(internal_entry["k_nnphi1"], dtype=np.float32)[:self.neighborhood_size],
                "k_nnphi2": np.array(internal_entry["k_nnphi2"], dtype=np.float32)[:self.neighborhood_size]
            }

            if self.dist_mode == 'rbf':
                knn_inf["k_nndist"] = self._rbf(knn_inf["k_nndist"]).astype(np.float32)
            elif self.dist_mode == 'exp':
                knn_inf["k_nndist"] = np.exp(- knn_inf["k_nndist"] / self.dist_mean)[:, np.newaxis]
            elif self.dist_mode == 'classic':
                knn_inf["k_nndist"] = knn_inf["k_nndist"][:, np.newaxis]  # modify at 2021/07/26

            if self.use_trigonometric:
                knn_inf["k_nnomega"] = self._torsion_triangle(knn_inf["k_nnomega"]).transpose(1, 0)
                knn_inf["k_nntheta1"] = self._torsion_triangle(knn_inf["k_nntheta1"]).transpose(1, 0)
                knn_inf["k_nntheta2"] = self._torsion_triangle(knn_inf["k_nntheta2"]).transpose(1, 0)
                knn_inf["k_nndelta1"] = self._torsion_triangle(knn_inf["k_nndelta1"]).transpose(1, 0)
                knn_inf["k_nndelta2"] = self._torsion_triangle(knn_inf["k_nndelta2"]).transpose(1, 0)
                knn_inf["k_nnphi1"] = self._torsion_triangle(knn_inf["k_nnphi1"]).transpose(1, 0)
                knn_inf["k_nnphi2"] = self._torsion_triangle(knn_inf["k_nnphi2"]).transpose(1, 0)
            else:
                knn_inf["k_nnomega"] = knn_inf["k_nnomega"][:, np.newaxis] / 180.0
                knn_inf["k_nntheta1"] = knn_inf["k_nntheta1"][:, np.newaxis] / 180.0
                knn_inf["k_nntheta2"] = knn_inf["k_nntheta2"][:, np.newaxis] / 180.0
                knn_inf["k_nndelta1"] = knn_inf["k_nndelta1"][:, np.newaxis] / 180.0
                knn_inf["k_nndelta2"] = knn_inf["k_nndelta2"][:, np.newaxis] / 180.0
                knn_inf["k_nnphi1"] = knn_inf["k_nnphi1"][:, np.newaxis] / 180.0
                knn_inf["k_nnphi2"] = knn_inf["k_nnphi2"][:, np.newaxis] / 180.0

        return label, cent_inf, knn_inf

    def __len__(self):
        return self.protein_list_len

    @staticmethod
    def _rbf(distance, num_rbf=16):
        """
        distance: input
        num_rbf: central bin
        """
        D_min, D_max, D_count = 0., 20., num_rbf
        D_mu = np.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape(-1, 1)
        D_sigma = (D_max - D_min) / D_count
        RBF = np.exp(-((distance - D_mu) / D_sigma) ** 2).transpose(1,0)
        return RBF

    @staticmethod
    def _torsion_triangle(torsion, k=1):
        """
        torsion: translate to triangle form
        k: cosine(k*(torsion))/sine(k*(torsion))
        """
        return np.array(list(map(lambda x: [np.cos(x * torsion), np.sin(x * torsion)], np.arange(1, k + 1))))[0]

    def _angle_triangle(self, input_crd, k=1):
        """
        angle: translate into Polar crd and triangle form
        """
        theta = np.arctan(input_crd[:, 1], input_crd[:, 0])
        cos_thta = self._torsion_triangle(theta, k).transpose(1, 0)
        rho = np.sqrt(input_crd[:, 0] ** 2 + input_crd[:, 1] ** 2)
        z = input_crd[:, 2]
        return np.concatenate([cos_thta, rho.reshape(-1, 1), z.reshape(-1, 1)], 1)
