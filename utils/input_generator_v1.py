import os
import jsonlines

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import PDBIO

from itertools import groupby
from collections import Counter
from functools import reduce

import pandas as pd
import numpy as np

from crd_relate import *
from gzmmcif_parser import *
from IO_relate import *
from protein_utils import *


def breakpointsearch(mmcif_file, chain, datatype = "gz"):

    if datatype == "gz":
        dict = GZMMCIF2Dict(mmcif_file)
    elif datatype == "mmCIF":
        from Bio.PDB.MMCIFParser import MMCIF2Dict
        dict = MMCIF2Dict(mmcif_file)

    new_dict = {"ATOM":dict["_atom_site.group_PDB"],"atom_id":dict["_atom_site.auth_seq_id"],"chain":dict["_atom_site.auth_asym_id"],
                "AA_type":dict["_atom_site.label_comp_id"], "icode":dict["_atom_site.pdbx_PDB_ins_code"],
                "Atom_type":dict['_atom_site.label_atom_id'], "X":dict["_atom_site.Cartn_x"], "Y":dict["_atom_site.Cartn_y"], "Z":dict["_atom_site.Cartn_z"],
                "altloc":dict['_atom_site.label_alt_id'], "model_num": dict['_atom_site.pdbx_PDB_model_num'], "bfactor": dict['_atom_site.B_iso_or_equiv']}
    df = pd.DataFrame.from_dict(new_dict)
    df = df[df["model_num"] == list(set(dict['_atom_site.pdbx_PDB_model_num']))[0]]

    mainatom = ["N", "CA", "C", "O"]

    altloclist = list(set(df[(df["chain"] == chain) & (df["ATOM"] == "ATOM")]["altloc"].tolist()))

    if "." in altloclist:
        atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(mainatom)) & (df["ATOM"] == "ATOM") & (df["altloc"] == ".") & (df["chain"] == chain) & (df["AA_type"] != "MSE")]
    else:
        atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(mainatom)) & (df["ATOM"] == "ATOM") & (df["altloc"] == altloclist[0]) & (df["chain"] == chain) & (df["AA_type"] != "MSE")]

    countlist = atomdf.iloc[:,[1,2]].to_numpy().tolist()
    countlist = list(map(lambda x:"".join(x),countlist))
    atomdf = atomdf.assign(atom_idchain=pd.Series(countlist).values)
    # from collections import Counter
    count = Counter(countlist)

    count4df = pd.DataFrame.from_records(np.array(list(count.items())))
    allless4atom = count4df[count4df.iloc[:,1] != "4"].iloc[:,0].to_numpy()
    filterdf = atomdf[~atomdf["atom_idchain"].isin(allless4atom)]

    seq_list = filterdf[filterdf["Atom_type"] == "N"]["atom_id"].astype(int).tolist()

    # from itertools import groupby
    tmp_length_list = []
    fun = lambda x: x[1] - x[0]
    for k, g in groupby(enumerate(seq_list), fun):
        l1 = [j for i, j in g]
        tmp_length_list.append(l1)

    breakchains = [i for i in tmp_length_list if len(i) > 2]

    filterdf["atom_id"] = filterdf["atom_id"].astype(int)
    filterdf = filterdf[filterdf["atom_id"].isin(np.concatenate(breakchains).tolist())]
    nodecrd = np.expand_dims(filterdf.iloc[:,6:9].to_numpy().astype(np.float32), 0)
    nodeseq = np.expand_dims(filterdf[filterdf["Atom_type"] == "N"]["AA_type"].to_numpy(), 0)
    nodebfactor = np.expand_dims(filterdf["bfactor"].to_numpy().astype(np.float32).reshape(-1,4).sum(1), 0)

    return breakchains, nodecrd, nodeseq, nodebfactor, dict


def sidepostprocess(crd, seq, bfactor, dihedral, exclude_onlyres = True):
    pad3seq = [np.pad(sing, (3,3), mode="constant", constant_values = "X") for sing in seq]
    pad3bfactor = [np.pad(sing, (3, 3), mode="constant", constant_values= -1.0) for sing in bfactor]
    dihedral = list(map(lambda x: list(map(lambda y: np.pad(y, (3,3), mode="constant", constant_values = 360) ,x)),dihedral))

    if exclude_onlyres == True:
        calseq = [i for i in range(len(seq)) if len(seq[i]) > 2]

    all_node_seq = []
    all_node_bfactor = []
    all_node_dih = []
    all_node_crd = []

    for i in calseq:
        seq_index = np.arange(np.array(pad3seq[i]).shape[0] - 6) + 3
        index = np.arange(np.array(pad3seq[i]).shape[0] - 6)

        all_index = list(map(lambda x: [x - 2, x - 1, x, x + 1, x + 2], seq_index))
        all_seqindex = list(map(lambda x: [x - 3, x - 2, x - 1, x, x + 1, x + 2, x + 3], seq_index))

        #### split node sequence
        all_node_seq.append(list(map(lambda x: list(map(lambda y: pad3seq[i][y], x)), all_seqindex)))
        all_node_bfactor.append(list(map(lambda x: list(map(lambda y: pad3bfactor[i][y], x)), all_seqindex)))

        #### split node dihedral angles
        sing_node_dih = []
        for dih in range(3):
            sing_node_dih.append(np.array(list(map(lambda x: list(map(lambda y: dihedral[dih][i][y], x)), all_index))))
        all_node_dih.append(sing_node_dih)

        #### split node central aa crd
        all_node_crd.append(list(map(lambda x: list(map(lambda y: crd[i][y], [4*x, 4*x+1, 4*x+2, 4*x+3])), index)))

    return all_node_seq, all_node_dih, all_node_crd, all_node_bfactor


def batch_generator(filename, pdbdataroot = "/Volumes/Fast_SSD/data20200127/Newdatafolder/datanew/", dsspdataroot = "/Volumes/Fast_SSD/divideddssp/", datatype = "mmCIF", outputroot = "/Volumes/Fast_SSD/BERTseq/data/0816test/pdbtest", inference = True, otherchain = True, knn = 20):
    if isinstance(filename, str):
        list_pdb = []
        list_chain = []
        with open(filename, 'r') as fil:
            for num_id, line in enumerate(fil.readlines()):
                list_pdb.append(line.strip().split("_")[0])
                list_chain.append(line.strip().split("_")[1])

    elif isinstance(filename, list):
        list_pdb = []
        list_chain = []
        for num_id, line in enumerate(filename):
            list_pdb.append(line.strip().split("_")[0])
            list_chain.append(line.strip().split("_")[1])

    all_datatype = datatype

    for i in range(len(list_pdb)):
        pdb = list_pdb[i]
        cur_chain = list_chain[i]
        # cur_dssp = os.path.join(dsspdataroot, pdb[1:3], pdb + ".dssp")
        cur_dssp = os.path.join(dsspdataroot, pdb + ".dssp")
        cur_cif = None

        if datatype == "gz":
            # cur_cif = os.path.join(pdbdataroot, pdb[1:3], pdb + ".cif.gz")
            cur_cif = os.path.join(pdbdataroot, pdb + ".cif.gz")
        elif datatype == "mmCIF" or datatype == "mmcif":
            # cur_cif = os.path.join(pdbdataroot, pdb[1:3], pdb + ".cif")
            cur_cif = os.path.join(pdbdataroot, pdb + ".cif")
        elif datatype == "PDB" or datatype == "pdb":

            cur_pdb = os.path.join(pdbdataroot, pdb + ".pdb")
            parser = PDBParser()
            structure = parser.get_structure(pdb, cur_pdb)

            if os.path.isdir(os.path.join(outputroot, pdb + "_" + cur_chain)):
                pass
            else:
                os.mkdir(os.path.join(outputroot, pdb + "_" + cur_chain))

            io = MMCIFIO()
            io.set_structure(structure)
            io.save(os.path.join(outputroot, pdb + "_" + cur_chain, pdb + ".cif"))
            cur_cif = os.path.join(outputroot, pdb + "_" + cur_chain, pdb + ".cif")
            datatype = "mmCIF"

        ### search breakpoint in main chain
        breakchains, nodecrd, nodeseq, nodebfactor, cif_dict = breakpointsearch(cur_cif, cur_chain, datatype=datatype)

        if inference == False:
            feasible_k1k2, feasible_atomid = read_sidechain(cif_dict, cur_chain, breakchain=breakchains, datatype=datatype)

        # ### extract SS & SASA
        if len(cur_chain) == 3:
            dsspchain = cur_chain[0]
        else:
            dsspchain = cur_chain

        if inference == False:
            ss8, ss3, rsa, newbreakchain = extract_SS_ASA_fromDSSP(cur_dssp, breakchains, dsspchain, includeside=True)

        ### transform aa from 3 letters to 1 letter
        nodeseq = transform_aa3to1(nodeseq)

        ### calculate phi, psi, omega
        dihedral = calc_dihedral(nodecrd, includeside=True)

        ### postprocess all of node
        node_seq, node_idh, node_crd, node_bfactor = sidepostprocess(nodecrd, nodeseq, nodebfactor, dihedral)

        ### calc distance
        k_nnidx, k_nndist, atomdf, CAdf, idx, knnAA, k_nn = calc_cif_interfacedistance(cif_dict, cur_chain, breakchains, datatype=datatype, otherchain=otherchain, k_nn=knn)

        ### calc angle, sphere
        knnangle, knnsphere3d = calc_angle_quaternion(idx, atomdf, CAdf, node_crd, k_nn)

        if inference == False:
            ## uniform old transform and save 2 json
            intertransform_sing_nodeedge(outputroot, pdb, cur_chain, breakchains,
                                         node_seq, node_idh, node_crd, node_bfactor, ss8, ss3, rsa, k_nnidx, k_nndist,
                                         knnAA, knnangle, knnsphere3d, feasible_k1k2, feasible_atomid, includeside=True,
                                         outputdistance=True)
        else:
            inferencewrite(outputroot, pdb, cur_chain, breakchains, node_seq,
                           node_idh, node_crd, node_bfactor, k_nnidx, k_nndist, knnAA, knnangle, knnsphere3d,
                           includeside=True, outputdistance=True)

        print(pdb + cur_chain + " has been processed!")

        datatype = all_datatype


def check_isfile(root, pdblist):
    def check_1pdb(root, pdbname):

        return (os.path.isfile(os.path.join(root, pdbname, "AAtype.jsonl")) and \
               os.path.isfile(os.path.join(root, pdbname, "angle.jsonl")) and \
               os.path.isfile(os.path.join(root, pdbname, "director.jsonl")) and \
               os.path.isfile(os.path.join(root, pdbname, "dist.jsonl")) and \
               os.path.isfile(os.path.join(root, pdbname, "pdbname.txt")) and \
               os.path.isfile(os.path.join(root, pdbname, "sphere3d.jsonl")))

    return np.array(pdblist)[~np.array(list(map(lambda pdb: check_1pdb(root, pdb), pdblist)))].tolist()



if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml
    import os

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    if os.path.isdir(config['preprocess_params']['outputroot']):
        pass
    else:
        os.mkdir(config['preprocess_params']['outputroot'])

    list_pdbchain = []

    with open(config["preprocess_params"]["filename"], 'r') as fil:
        for num_id, line in enumerate(fil.readlines()):
            list_pdbchain.append(line.strip())

    no_feat_pdb = check_isfile(config["preprocess_params"]["outputroot"], list_pdbchain)

    batch_generator(filename=no_feat_pdb, pdbdataroot = config["preprocess_params"]["pdbdataroot"], datatype = config["preprocess_params"]["datatype"],
                    outputroot = config["preprocess_params"]["outputroot"], inference = config["preprocess_params"]["inference"], otherchain = config["preprocess_params"]["otherchain"])

    ### check
    # batch_generator(filename="/home/liuyf/alldata/experiment/TIM/batchtest.txt", pdbdataroot = "/home/liuyf/alldata/experiment/TIM/PDBfile", datatype="PDB", inference=True, outputroot = "/home/liuyf/alldata/experiment/TIM/inference_input", otherchain= True)
