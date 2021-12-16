import os
import sys
import json
sys.path.append("../utils")
import numpy as np

from torch.utils import data
import torch


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


class ChainDataset(data.Dataset):
    def __init__(self, dirroot, neighborhood_size, protein_list=None,
                 use_normalize=False, use_rbf=False, use_trigonometric=False, old_features=True):
        self.dirroot = dirroot
        self.neighborhood_size = neighborhood_size
        self.use_normalize = use_normalize
        self.use_rbf = use_rbf
        self.old_features = old_features
        self.use_trigonometric = use_trigonometric
        self.all_data = []

        self.dist_mean = 7.77
        self.rsa_mean_std = [44.290062, 45.782705]  # calculated from trainset only
        self.bfactor_mean_std = [114.900829, 74.134204]  # calculated from trainset only

        if protein_list is not None:
            with open(protein_list, "r") as f:
                protein_list = f.readlines()
                protein_list = [x.strip() for x in protein_list]
        else:
            protein_list = ['1r26']

        self.protein_list = protein_list

        self.protein_list_len = len(self.protein_list)
        print(self.protein_list_len)

    def __getitem__(self, index):
        protein = self.protein_list[index]

        node_file = os.path.join(self.dirroot, protein, "pdbname.txt")
        director_file = os.path.join(self.dirroot, protein, "director.jsonl")
        angle_file = os.path.join(self.dirroot, protein, "angle.jsonl")
        AAtype_file = os.path.join(self.dirroot, protein, "AAtype.jsonl")
        dist_file = os.path.join(self.dirroot, protein, "dist.jsonl")
        sphere3d_file = os.path.join(self.dirroot, protein, "sphere3d.jsonl")
        allinternal_file = os.path.join(self.dirroot, protein, "split_all_internal.jsonl")

        with open(node_file) as nodf:
            nodflines = nodf.readlines()
        with open(director_file) as dirf:
            dirflines = dirf.readlines()

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

        node_list = [x.strip() for x in nodflines]
        node_dict = {}
        for i, node in enumerate(node_list):
            node_dict[node.split("+")[0].split("_")[1] + node.split("+")[1]] = i

        all_seq = []
        all_cent_inf = []
        all_knn_inf = []
        all_otherchain = []

        for node in range(len(node_list)):
            entry = json.loads(dirflines[node])
            if self.old_features:
                angle_entry = json.loads(anflines[node])
                dist_entry = json.loads(disflines[node])
                sphere_entry = json.loads(sphflines[node])
                aa_entry = json.loads(aaflines[node])
            else:
                internal_entry = json.loads(itflines[node])

            # AAtype
            all_seq.append(entry['centralAA'])

            # cent inf
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

            cent_inf = {
                "pdbname": entry["pdbname"],
                "node_dihedral": node_diherdral
            }

            if self.use_rbf:
                cent_inf["dist"] = self._rbf(np.array([0.0])).astype(np.float32)
            else:
                cent_inf["dist"] = np.array([1.0], dtype=np.float32)[:, np.newaxis]

            all_otherchain.extend(np.array([[node, colidx, aa_entry["knnAAtype"][colidx]] if knnnode[0] + knnnode[1] not in node_dict else [-1,-1, -1]
                                                   for colidx, knnnode in enumerate(entry["k_nnidx"])])[:self.neighborhood_size])

            # knn inf
            if self.old_features:
                knn_inf = {
                    "knnind": knnind(entry["k_nnidx"], entry["pdbname"])[:self.neighborhood_size],
                    "knnpos": np.array([node_dict[knnnode[0] + knnnode[1]] if knnnode[0] + knnnode[1] in node_dict
                                        else -1 for knnnode in entry["k_nnidx"]])[:self.neighborhood_size],
                    "k_nndist": np.array(dist_entry["k_nndist"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nnangle": np.array(angle_entry["k_nnangle"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nnsphere3d": np.array(sphere_entry["k_nnsphere3d"], dtype=np.float32)[:self.neighborhood_size]
                }

                if self.use_rbf:
                    knn_inf["k_nndist"] = self._rbf(knn_inf["k_nndist"]).astype(np.float32)
                else:
                    # knn_inf["k_nndist"] = knn_inf["k_nndist"][:, np.newaxis]  # modify at 2021/07/26
                    knn_inf["k_nndist"] = np.exp(- knn_inf["k_nndist"] / self.dist_mean)[:, np.newaxis]

                if self.use_trigonometric:
                    knn_inf["k_nnangle"] = self._angle_triangle(knn_inf["k_nnangle"])
                    knn_inf["k_nnsphere3d"] = self._angle_triangle(knn_inf["k_nnsphere3d"])

                # knn_inf["k_nnangle"] = knn_inf["k_nnangle"] / np.expand_dims(
                #     np.sqrt((knn_inf["k_nnangle"] ** 2).sum(1)), 1)

            else:
                knn_inf = {
                    "knnind": knnind(internal_entry["k_nnid"], entry["pdbname"])[:self.neighborhood_size],
                    "knnpos": np.array([node_dict[knnnode[0] + knnnode[1]] if knnnode[0] + knnnode[1] in node_dict
                                        else -1 for knnnode in entry["k_nnidx"]])[:self.neighborhood_size],
                    "k_nndist": np.array(internal_entry["k_nndist"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nnomega": np.array(internal_entry["k_nnomega"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nntheta1": np.array(internal_entry["k_nntheta1"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nntheta2": np.array(internal_entry["k_nntheta2"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nndelta1": np.array(internal_entry["k_nndelta1"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nndelta2": np.array(internal_entry["k_nndelta2"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nnphi1": np.array(internal_entry["k_nnphi1"], dtype=np.float32)[:self.neighborhood_size],
                    "k_nnphi2": np.array(internal_entry["k_nnphi2"], dtype=np.float32)[:self.neighborhood_size]
                }

                if self.use_rbf:
                    knn_inf["k_nndist"] = self._rbf(knn_inf["k_nndist"]).astype(np.float32)
                else:
                    # knn_inf["k_nndist"] = knn_inf["k_nndist"][:, np.newaxis]  # modify at 2021/07/26
                    knn_inf["k_nndist"] = np.exp(- knn_inf["k_nndist"] / self.dist_mean)[:, np.newaxis]

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

            all_cent_inf.append(cent_inf)
            all_knn_inf.append(knn_inf)

            # if self.pred_k1k2:
            #     all_k1k2.append(k1k2_entry["k_nndist"])

        chain_seq = np.array(all_seq)
        chain_cent_inf = {
            "pdbname": [cent_inf["pdbname"] for cent_inf in all_cent_inf],
            "node_dihedral" : {
                'phi': np.array([cent_inf["node_dihedral"]['phi'] for cent_inf in all_cent_inf]),
                'psi': np.array([cent_inf["node_dihedral"]['psi'] for cent_inf in all_cent_inf]),
                'omega': np.array([cent_inf["node_dihedral"]['omega'] for cent_inf in all_cent_inf]),
            },
            "dist": np.array([cent_inf["dist"] for cent_inf in all_cent_inf])
        }

        # if self.pred_k1k2:
        #     chain_cent_inf["k1k2"] = np.array(all_k1k2, dtype=np.float32) / 180.0
        #     chain_cent_inf["k1k2_mask"] = (chain_cent_inf["k1k2"] >= -1.0) & (chain_cent_inf["k1k2"] <= 1.0)

        chain_knn_inf = {}
        for key in list(all_knn_inf[0].keys()):
            chain_knn_inf[key] = np.array([knn_inf[key] for knn_inf in all_knn_inf])

        chain_cent_inf["otherchain_idx"] = np.array(all_otherchain)[np.array(all_otherchain) > -1].reshape(-1,3)

        return chain_seq, chain_cent_inf, chain_knn_inf

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
        k: cosine(k*(torsion)) and sine(k*(torsion))
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

