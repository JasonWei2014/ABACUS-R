import os
import jsonlines

import pandas as pd
import numpy as np

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import make_dssp_dict

from gzmmcif_parser import *
from protein_utils import *

def parse_dssp_from_dict(dssp_file):
    ## 1a2f.dssp
    d = make_dssp_dict(dssp_file)
    appender = []
    for k in d[1]:
        to_append = []
        y = d[0][k]
        chain = k[0]
        residue = k[1]
        het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    cols = ['chain','resnum', 'icode' ,'aa', 'ss', 'exposure_rsa', 'phi', 'psi','ggg',
            'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx',
            'O_NH_1_energy', 'NH_O_2_relidx', 'NH_O_2_energy',
            'O_NH_2_relidx', 'O_NH_2_energy']

    df = pd.DataFrame.from_records(appender, columns=cols)

    return df

def extract_SS_ASA_fromDSSP(dssp_file, breakchains, chain, includeside = False):
    df = parse_dssp_from_dict(dssp_file)
    if includeside == False:
        break_idx = list(map(lambda x: x[3:-3], breakchains))
    else:
        break_idx = list(map(lambda x: x, breakchains))

    break_idx = np.concatenate(break_idx, 0).astype(np.int).tolist()

    df = df[(df["chain"] == chain) & (df["resnum"].isin(break_idx)) & (df["icode"] == " ")]

    ss8_series = df['ss']
    ss3_series = ss8_series.copy()
    rsa_series = df['exposure_rsa']

    ss3_series.loc[(ss8_series == 'T')|(ss8_series == 'S')|(ss8_series == '-')] = "L"
    ss3_series.loc[(ss8_series == 'H') | (ss8_series == 'G') | (ss8_series == 'I')] = "H"
    ss3_series.loc[(ss8_series == 'B') | (ss8_series == 'E')] = "E"

    return ss8_series.tolist(), ss3_series.tolist(), rsa_series.tolist(), df["resnum"].tolist()


def read_sidechain(mmcif_dict, chain, breakchain = None, datatype = "gz"):

    # if datatype == "gz":
    #     dict = GZMMCIF2Dict(mmcif_file)
    # elif datatype == "mmCIF":
    #     from Bio.PDB.MMCIFParser import MMCIF2Dict
    #     dict = MMCIF2Dict(mmcif_file)

    dict = mmcif_dict

    new_dict = {"ATOM":dict["_atom_site.group_PDB"],"atom_id":dict["_atom_site.auth_seq_id"],"chain":dict["_atom_site.auth_asym_id"],
                "AA_type":dict["_atom_site.label_comp_id"], "icode":dict["_atom_site.pdbx_PDB_ins_code"],
                "Atom_type":dict['_atom_site.label_atom_id'], "X":dict["_atom_site.Cartn_x"], "Y":dict["_atom_site.Cartn_y"], "Z":dict["_atom_site.Cartn_z"],
                "altloc":dict['_atom_site.label_alt_id'], "model_num": dict['_atom_site.pdbx_PDB_model_num']}
    df = pd.DataFrame.from_dict(new_dict)
    df = df[df["model_num"] == list(set(dict['_atom_site.pdbx_PDB_model_num']))[0]]

    sideatom = ["N", "CA", "CB", "CG", "CG1", "OG1", "OG", "SG", "CD", "CD1", "SD", "OD1", "ND1"]

    altloclist = list(set(df[(df["chain"] == chain) & (df["ATOM"] == "ATOM")]["altloc"].tolist()))

    if "." in altloclist:
        altloclist.sort()
        altloclist.remove(".")
        if len(altloclist) == 0:
            atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(sideatom)) & (df["ATOM"] == "ATOM") & (
                        df["altloc"] == ".") & (df["chain"] == chain) & (df["AA_type"] != "MSE")]
        else:
            atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(sideatom)) & (df["ATOM"] == "ATOM") & (
                        df["altloc"].isin([altloclist[0], "."])) & (df["chain"] == chain) & (df["AA_type"] != "MSE")]
    else:
        altloclist.sort()
        atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(sideatom)) & (df["ATOM"] == "ATOM") & (
                    df["altloc"] == altloclist[0]) & (df["chain"] == chain) & (df["AA_type"] != "MSE")]
    # chi1: c, ca, cb, cg
    atomdf["atom_id"] = atomdf["atom_id"].astype(np.int)
    if breakchain is not None:
        atomdf = atomdf[atomdf["atom_id"].isin(np.concatenate(breakchain).tolist())]
    def read_sideinf(atom_id, altloclist, atomdf):
        chi_dict = {
            "ARG": {"chi_1": "CG", "chi_2": "CD", "chi_3": "NE", "chi_4": "CZ"},
            "LYS": {"chi_1": "CG", "chi_2": "CD", "chi_3": "CE", "chi_4": "NZ"},
            "GLN": {"chi_1": "CG", "chi_2": "CD", "chi_3": "OE1"},
            "GLU": {"chi_1": "CG", "chi_2": "CD", "chi_3": "OE1"},
            "MET": {"chi_1": "CG", "chi_2": "SD", "chi_3": "CE"},
            "ASP": {"chi_1": "CG", "chi_2": "OD1"},
            "ILE": {"chi_1": "CG1", "chi_2": "CD1"},
            "HIS": {"chi_1": "CG", "chi_2": "ND1"},
            "LEU": {"chi_1": "CG", "chi_2": "CD1"},
            "ASN": {"chi_1": "CG", "chi_2": "OD1"},
            "PHE": {"chi_1": "CG", "chi_2": "CD1"},
            "PRO": {"chi_1": "CG", "chi_2": "CD"},
            "TRP": {"chi_1": "CG", "chi_2": "CD1"},
            "TYR": {"chi_1": "CG", "chi_2": "CD1"},
            "VAL": {"chi_1": "CG1"},
            "THR": {"chi_1": "OG1"},
            "SER": {"chi_1": "OG"},
            "CYS": {"chi_1": "SG"},
            "GLY": {},
            "ALA": {},
            "UNK": {},
            "PYL": {"chi_1": "CG", "chi_2": "CD", "chi_3": "CE", "chi_4": "NZ"},
            "SEC": {"chi_1": "SG"},
            "XLE": {"chi_1": "CG", "chi_2": "CD1"},
            "GLX": {"chi_1": "CG", "chi_2": "CD", "chi_3": "OE1"},
            "XAA": {},
            "ASX": {"chi_1": "CG", "chi_2": "OD1"}
        }
        non_standardAA = {"ASX": "ASP", "XAA": "GLY", "GLX": "GLU", "XLE": "LEU", "SEC": "CYS", "PYL": "LYS", "UNK": "GLY"}

        miss_list = []
        chi_list = []

        atomdf['AA_type'].replace(non_standardAA, inplace=True)
        if chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]].__contains__("chi_1"):
            if chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]].__contains__("chi_2"):
                if set(["N", "CA", "CB"] + [chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"]] +
                    [chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_2"]]).issubset(set(atomdf[atomdf["atom_id"] == atom_id].iloc[:, 5].tolist())):

                    if atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "GLY" or atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "ALA":
                        chi_list.append([360. ,360. ])
                    else:
                        sing_chi_list = []
                        n_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "N")].iloc[:,
                                6:9].to_numpy().astype(np.float)

                        ca_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CA")].iloc[:,
                                6:9].to_numpy().astype(np.float)
                        cb_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CB")].iloc[:,
                                6:9].to_numpy().astype(np.float)

                        ## if "." not "A"
                        try:
                            cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (atomdf["Atom_type"] ==
                                                                          chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"])].iloc[:,
                                6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))

                        except:
                            cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (atomdf["Atom_type"] ==
                                                                          chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"])].iloc[:,
                                6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))


                        if chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]].__contains__("chi_2"):
                            ## if "." not "A"
                            try:
                                cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (atomdf["Atom_type"] ==
                                                                          chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_2"])].iloc[:,
                                6:9].to_numpy().astype(np.float)
                                sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))

                            except:
                                cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (atomdf["Atom_type"] ==
                                                                          chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_2"])].iloc[:,
                                6:9].to_numpy().astype(np.float)
                                sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))


                        else:
                            sing_chi_list.append(0.)

                        chi_list.append(sing_chi_list)
                        # miss_list.append(-100)
                else:
                    miss_list.append(int(atomdf[atomdf["atom_id"] == atom_id].iloc[0, 1]))
                    chi_list.append([0., 0.])
            else:
                if set(["N", "CA", "CB"] + [chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"]]).issubset(
                    set(atomdf[atomdf["atom_id"] == atom_id].iloc[:, 5].tolist())):

                    if atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "GLY" or \
                            atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "ALA":
                        chi_list.append([360., 360.])
                    else:
                        sing_chi_list = []
                        n_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "N")].iloc[:,
                                6:9].to_numpy().astype(np.float)

                        ca_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CA")].iloc[:,
                                 6:9].to_numpy().astype(np.float)
                        cb_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CB")].iloc[:,
                                 6:9].to_numpy().astype(np.float)

                        ## if "." not "A"
                        try:
                            cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (
                                        atomdf["Atom_type"] ==
                                        chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"])].iloc[:,
                                     6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))

                        except:
                            cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (
                                        atomdf["Atom_type"] ==
                                        chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"])].iloc[:,
                                     6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))

                        if chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]].__contains__("chi_2"):
                            ## if "." not "A"
                            try:
                                cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (
                                            atomdf["Atom_type"] ==
                                            chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]][
                                                "chi_2"])].iloc[:,
                                         6:9].to_numpy().astype(np.float)
                                sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))

                            except:
                                cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (
                                            atomdf["Atom_type"] ==
                                            chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]][
                                                "chi_2"])].iloc[:,
                                         6:9].to_numpy().astype(np.float)
                                sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))


                        else:
                            sing_chi_list.append(0.)

                        chi_list.append(sing_chi_list)
                        # miss_list.append(-100)
                else:
                    miss_list.append(int(atomdf[atomdf["atom_id"] == atom_id].iloc[0, 1]))
                    chi_list.append([0., 0.])
        else:
            if set(["N", "CA"]).issubset(
                set(atomdf[atomdf["atom_id"] == atom_id].iloc[:, 5].tolist())):

                if atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "GLY" or \
                        atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3] == "ALA":
                    chi_list.append([360., 360.])
                else:
                    sing_chi_list = []
                    n_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "N")].iloc[:,
                            6:9].to_numpy().astype(np.float)

                    ca_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CA")].iloc[:,
                             6:9].to_numpy().astype(np.float)
                    cb_crd = atomdf[(atomdf["atom_id"] == atom_id) & (atomdf["Atom_type"] == "CB")].iloc[:,
                             6:9].to_numpy().astype(np.float)

                    ## if "." not "A"
                    try:
                        cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (atomdf["Atom_type"] ==
                                                                                                  chi_dict[atomdf[
                                                                                                      atomdf[
                                                                                                          "atom_id"] == atom_id].iloc[
                                                                                                      0, 3]][
                                                                                                      "chi_1"])].iloc[:,
                                 6:9].to_numpy().astype(np.float)
                        sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))

                    except:
                        cg_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (
                                    atomdf["Atom_type"] ==
                                    chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_1"])].iloc[:,
                                 6:9].to_numpy().astype(np.float)
                        sing_chi_list.append(torsion(n_crd[0], ca_crd[0], cb_crd[0], cg_crd[0]))

                    if chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]].__contains__("chi_2"):
                        ## if "." not "A"
                        try:
                            cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == ".") & (
                                        atomdf["Atom_type"] ==
                                        chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_2"])].iloc[:,
                                     6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))

                        except:
                            cd_crd = atomdf[(atomdf["atom_id"] == atom_id) & (df["altloc"] == altloclist[0]) & (
                                        atomdf["Atom_type"] ==
                                        chi_dict[atomdf[atomdf["atom_id"] == atom_id].iloc[0, 3]]["chi_2"])].iloc[:,
                                     6:9].to_numpy().astype(np.float)
                            sing_chi_list.append(torsion(ca_crd[0], cb_crd[0], cg_crd[0], cd_crd[0]))


                    else:
                        sing_chi_list.append(0.)

                    chi_list.append(sing_chi_list)

            else:
                miss_list.append(int(atomdf[atomdf["atom_id"] == atom_id].iloc[0, 1]))
                chi_list.append([0., 0.])
        if len(miss_list) == 0:
            miss_list.append(-100)
        return [chi_list, miss_list, atom_id]

    all_k1k2= list(map(lambda x: read_sideinf(x, altloclist, atomdf), sorted(set(atomdf["atom_id"].tolist()))))
    loss_k1k2 = np.array(list(map(lambda x:x[0], all_k1k2)))
    miss_list = list(map(lambda x: x[1], all_k1k2))
    atom_idlist = list(map(lambda x: x[2], all_k1k2))

    feasible_atomid = np.array(atom_idlist)[np.where(np.array(miss_list)<0)[0]]
    feasible_k1k2 = np.array(loss_k1k2)[np.where(np.array(miss_list)<0)[0]]

    return feasible_k1k2.squeeze(1), feasible_atomid


def writetxt(pdblist, outputfile):
    with open(outputfile, 'w') as fil:
        for pdb in pdblist:
            fil.write(pdb + "\n")

def check4empty(node_seq, node_bfactor, node_idh, node_crd):
    check_list = [False if len(subchain) == 0 else True for subchain in node_seq]

    if np.all(np.array(check_list)) == False:
        true_list = np.nonzero(check_list)[0]
        node_seq = list(map(lambda x: node_seq[x], true_list))
        node_bfactor = list(map(lambda x: node_bfactor[x], true_list))
        node_idh = list(map(lambda x: node_idh[x], true_list))
        node_crd = list(map(lambda x: node_crd[x], true_list))

    return node_seq, node_bfactor, node_idh, node_crd

def intertransform_sing_nodeedge(outputname, pdb, chain, breakchains, node_seq, node_idh, node_crd, node_bfactor, ss8, ss3, rsa, k_nnidx, k_nndist, knnAA, knnangle, knnsphere3d, feasible_k1k2, feasible_atomid, includeside = False, encodeseq = True, encodess = True, outputdistance = False):

    #### chaeck dim
    node_seq, node_bfactor, node_idh, node_crd = check4empty(node_seq, node_bfactor, node_idh, node_crd)
    # #### node feature

    if includeside == False:
        break_idx = list(map(lambda x: x[3:-3], breakchains))
    else:
        break_idx = list(map(lambda x: x, breakchains))
    break_idx = np.concatenate(break_idx, 0).astype(np.int).tolist()
    seqnearpdbidx = list(map(lambda y: [y-3,y-2,y-1,y,y+1,y+2,y+3], break_idx))

    if encodeseq == True:
        seq_df = pd.DataFrame.from_records(np.concatenate(node_seq, 0))
        seq_df.replace(ENCODEAA2NUM,inplace=True)
        centtype = seq_df.to_numpy()[:, 3].tolist()
        node_seq = seq_df.to_numpy().tolist()
    else:
        centtype = np.concatenate(node_seq, 0)[:,3].tolist()
        node_seq = np.concatenate(node_seq, 0).tolist()

    if encodess == True:
        ss8df = pd.Series(ss8)
        ss3df = pd.Series(ss3)
        ss8df.replace(ENCODESS82NUM, inplace=True)
        ss3df.replace(ENCODESS32NUM, inplace=True)
        ss8 = ss8df.tolist()
        ss3 = ss3df.tolist()

    node_bfactor = np.concatenate(node_bfactor, 0).tolist()
    node_idh = np.concatenate(node_idh,1).transpose([1,0,2]).tolist()
    node_crd = np.concatenate(node_crd, 0).tolist()

    k_nnidx = k_nnidx.tolist()
    k_nndist = k_nndist.tolist()
    feasible_k1k2 = feasible_k1k2.tolist()
    feasible_atomid = feasible_atomid.tolist()

    if os.path.isdir(os.path.join(outputname, pdb + "_" + chain)):
        pass
    else:
        os.mkdir(os.path.join(outputname, pdb + "_" + chain))

    pdbresname = list(map(lambda x: pdb + "_" + chain + "+" + str(x), break_idx))
    writetxt(pdbresname, os.path.join(outputname, pdb + "_" + chain, "pdbname.txt"))

    import jsonlines
    with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "director.jsonl"), mode="a") as writer:
        for dictidx in range(np.array(node_seq).shape[0]):
            eachdic = {"pdbname":pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                       "centralAA": centtype[dictidx],
                       "ss8": ss8[dictidx],
                       "ss3": ss3[dictidx],
                       "rsa": rsa[dictidx],
                       "nodeseq": node_seq[dictidx],
                       "nodeseq_id": seqnearpdbidx[dictidx],
                       "nodebfactor":node_bfactor[dictidx][3],
                       "node_dihedral": {"phi": node_idh[dictidx][0],
                                         "psi": node_idh[dictidx][1],
                                         "omega": node_idh[dictidx][2]},
                       "node_crd_mainchain": node_crd[dictidx],
                       "k_nnidx": k_nnidx[dictidx]
                       }
            writer.write(eachdic)
    print("Write pointer success!")
    if outputdistance == True:
        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "dist.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname":pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nndist": k_nndist[dictidx]
                           }
                writer.write(eachdic)
        print("Write distance success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "AAtype.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "knnAAtype": knnAA[dictidx]
                           }
                writer.write(eachdic)
        print("Write AAtype success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "angle.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nnangle": knnangle[dictidx]
                           }
                writer.write(eachdic)
        print("Write angle success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "sphere3d.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nnsphere3d": knnsphere3d[dictidx]
                           }
                writer.write(eachdic)
        print("Write sphere3d success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "k1k2.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                res_id = break_idx[dictidx]
                if res_id in feasible_atomid:
                    k1k2index = feasible_atomid.index(res_id)
                    k1k2angle = feasible_k1k2[k1k2index]
                else:
                    k1k2angle = [360.0, 360.0]

                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "feasible_k1k2": k1k2angle
                           }
                writer.write(eachdic)
        print("Write k1k2 success!")

def inferencewrite(outputname, pdb, chain, breakchains, node_seq, node_idh, node_crd, node_bfactor, k_nnidx, k_nndist, knnAA, knnangle, knnsphere3d, includeside = False, encodeseq = True, outputdistance = False):

    #### check dim
    node_seq, node_bfactor, node_idh, node_crd = check4empty(node_seq, node_bfactor, node_idh, node_crd)

    if includeside == False:
        break_idx = list(map(lambda x: x[3:-3], breakchains))
    else:
        break_idx = list(map(lambda x: x, breakchains))

    break_idx = np.concatenate(break_idx, 0).astype(np.int32).tolist()
    seqnearpdbidx = list(map(lambda y: [y-3,y-2,y-1,y,y+1,y+2,y+3], break_idx))

    if encodeseq == True:
        seq_df = pd.DataFrame.from_records(np.concatenate(node_seq, 0))
        seq_df.replace(ENCODEAA2NUM,inplace=True)
        centtype = seq_df.to_numpy()[:, 3].tolist()
        node_seq = seq_df.to_numpy().tolist()
    else:
        centtype = np.concatenate(node_seq, 0)[:,3].tolist()
        node_seq = np.concatenate(node_seq, 0).tolist()


    node_bfactor = np.concatenate(node_bfactor, 0).tolist()
    node_idh = np.concatenate(node_idh,1).transpose([1,0,2]).tolist()
    node_crd = np.concatenate(node_crd, 0).tolist()

    k_nnidx = k_nnidx.tolist()
    k_nndist = k_nndist.tolist()

    if os.path.isdir(os.path.join(outputname, pdb + "_" + chain)):
        pass
    else:
        os.mkdir(os.path.join(outputname, pdb + "_" + chain))

    pdbresname = list(map(lambda x: pdb + "_" + chain + "+" + str(x), break_idx))
    writetxt(pdbresname, os.path.join(outputname, pdb + "_" + chain, "pdbname.txt"))

    import jsonlines
    with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "director.jsonl"), mode="a") as writer:
        for dictidx in range(np.array(node_seq).shape[0]):
            eachdic = {"pdbname":pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                       "centralAA": centtype[dictidx],
                       "nodeseq": node_seq[dictidx],
                       "nodeseq_id": seqnearpdbidx[dictidx],
                       "nodebfactor":node_bfactor[dictidx][3],
                       "node_dihedral": {"phi": node_idh[dictidx][0],
                                         "psi": node_idh[dictidx][1],
                                         "omega": node_idh[dictidx][2]},
                       "node_crd_mainchain": node_crd[dictidx],
                       "k_nnidx": k_nnidx[dictidx]
                       }
            writer.write(eachdic)
    print("Write pointer success!")
    if outputdistance == True:
        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "dist.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname":pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nndist": k_nndist[dictidx]
                           }
                writer.write(eachdic)
        print("Write distance success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "AAtype.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "knnAAtype": knnAA[dictidx]
                           }
                writer.write(eachdic)
        print("Write AAtype success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "angle.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nnangle": knnangle[dictidx]
                           }
                writer.write(eachdic)
        print("Write angle success!")

        with jsonlines.open(os.path.join(outputname, pdb + "_" + chain, "sphere3d.jsonl"), mode="a") as writer:
            for dictidx in range(np.array(node_seq).shape[0]):
                eachdic = {"pdbname": pdb + "_" + chain + "+" + str(break_idx[dictidx]),
                           "k_nnsphere3d": knnsphere3d[dictidx]
                           }
                writer.write(eachdic)
        print("Write sphere3d success!")