from collections import Counter

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from protein_utils import *

def quaternion_transform(r):
    """
    Get optimal rotation
    note: translation will be zero when the centroids of each molecule are the
    same
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot

def makeW(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    W = np.asarray([
        [r4, r3, -r2, r1],
        [-r3, r4, r1, r2],
        [r2, -r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return W

def makeQ(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    Q = np.asarray([
        [r4, -r3, r2, r1],
        [r3, r4, -r1, r2],
        [-r2, r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return Q

def quaternion_rotate(X, Y):
    """
    Calculate the rotation

    Parameters
    ----------
    X : numpy.array
        (N,D) matrix, where N is points and D is dimension.
    Y: numpy.array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rot : matrix
        Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])

    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot

def quaternion2rotmat(q0, q1, q2, q3):

    rotmat = np.asarray([[1 - (2*q2**2) - (2*q3**2), 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                        [2 * q1 * q2 + 2 * q0 * q3, 1 - (2*q1**2) - (2*q3**2), 2 * q2 * q3 - 2 * q0 * q1],
                        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - (2*q1**2) - (2*q2**2)]])
    return rotmat

def rotmat2quaternion(mat):
    q0 = np.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2] + 1e-6) / 2
    q1 = (mat[2, 1] - mat[1, 2]) / (4 * q0)
    q2 = (mat[0, 2] - mat[2, 0]) / (4 * q0)
    q3 = (mat[1, 0] - mat[0, 1]) / (4 * q0)
    return np.array([q0, q1, q2, q3])

def rotmat2theta(rotmat):
    return np.arccos(np.trace(rotmat - 1)/2)

def quaternion3d(a, b):
    rotmat = quaternion_rotate(a, b)
    quat = rotmat2quaternion(rotmat)
    quat_vector = quat[1:]
    cos = np.clip(quat[0], -1, 1)
    my_angle = 2 * np.arccos(cos)
    norm = np.linalg.norm(quat_vector, axis=0, keepdims=True)
    rotvector = quat_vector/norm
    return rotvector * my_angle

def rotaxis2m(theta, vector):

    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    x, y, z = vector[0],vector[1],vector[2]
    rot = np.zeros((3, 3))
    # 1st row
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y + s * z
    rot[0, 2] = t * x * z - s * y
    # 2nd row
    rot[1, 0] = t * x * y - s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z + s * x
    # 3rd row
    rot[2, 0] = t * x * z + s * y
    rot[2, 1] = t * y * z - s * x
    rot[2, 2] = t * z * z + c
    return rot

def trans_mtx(x):
    """
    [x,y,z,1].dot(trans_mtx)
    """
    return np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (x[0], x[1], x[2], 1)), dtype=np.float64)

def angle(x1, x2, x3):
    """
    calc_angle of point(x1), point(x2), point(x3)
    """
    ba = x1 - x2
    ba /= np.linalg.norm(ba, keepdims=True)
    bc = x3 - x2
    bc /= np.linalg.norm(bc, keepdims=True)
    cosine_angle = np.dot(ba,bc)
    angle1 = np.degrees(np.arccos(cosine_angle))
    angle2 = np.arccos(cosine_angle)
    return angle1, angle2

def rottrans(central_aa):
    #### transpose to (0,0,0)
    central_aa = np.concatenate((central_aa, np.ones((central_aa.shape[0], 1))), axis=1)
    transMat = trans_mtx(-central_aa[1])
    central_aa = central_aa.dot(transMat)  # translation
    central_aa = central_aa[:, :3]
    #### fix X axis
    _, xtmp_ang = angle(np.array([0.0, central_aa[0, 1],
                                      central_aa[0, 2]]),
                            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    if central_aa[0, 1] < 0:
        rotMat1 = rotaxis2m(-xtmp_ang, [1, 0, 0])
        central_aa = central_aa.dot(rotMat1)  # fix_X_rotation
    else:
        rotMat1 = rotaxis2m(xtmp_ang, [1, 0, 0])
        central_aa = central_aa.dot(rotMat1)

    #### fix Y axis
    _, ytmp_ang = angle(np.array([central_aa[0, 0], 0.0, central_aa[0, 2]]),
                                np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    if central_aa[0, 2] < 0:
        rotMat2 = rotaxis2m(-ytmp_ang, [0, 1, 0])
        central_aa = central_aa.dot(rotMat2)  # fix_Y_rotation
    else:
        rotMat2 = rotaxis2m(ytmp_ang, [0, 1, 0])
        central_aa = central_aa.dot(rotMat2)

    #### fix X axis
    _, xtmp_ang = angle(np.array([0.0, central_aa[2, 1],
                                      central_aa[2, 2]]),
                            np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    if central_aa[2, 2] > 0:
        rotMat3 = rotaxis2m(-xtmp_ang, [1, 0, 0])
        central_aa = central_aa.dot(rotMat3)  # fix_X_rotation
    else:
        rotMat3 = rotaxis2m(xtmp_ang, [1, 0, 0])
        central_aa = central_aa.dot(rotMat3)

    return central_aa, transMat, rotMat1.dot(rotMat2).dot(rotMat3)

def torsion(x1, x2, x3, x4, degrees = True):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0*b1, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x

    x = np.sum(v*w)
    b1xv = np.cross(b1, v)
    y = np.sum(b1xv*w)
    if degrees:
        return np.float32(180.0 / np.pi) * np.arctan2(y, x)
    else:
        return np.arctan2(y, x)


def calc_angle_quaternion(k_nnidx, atomdf, CAdf, node_crd, k_nn):

    rawidx = list((list(map(lambda x: CAdf.iloc[x, 0].values.tolist(), k_nnidx))))

    rawidxshape = np.array(rawidx).shape

    tmp_raw = np.concatenate(rawidx)

    atomdfidx = list(map(lambda x: [atomdf.index[atomdf.index.get_loc(x)-1],
                              atomdf.index[atomdf.index.get_loc(x)],
                              atomdf.index[atomdf.index.get_loc(x)+1],
                              atomdf.index[atomdf.index.get_loc(x)+2]],tmp_raw))

    tmp_idx = np.concatenate(atomdfidx)

    all_knncrd = np.array(list(zip(list(map(lambda x: atomdf.at[x,"X"], tmp_idx)),
                          list(map(lambda x: atomdf.at[x,"Y"], tmp_idx)),
                          list(map(lambda x: atomdf.at[x,"Z"], tmp_idx))))).astype(float).reshape([rawidxshape[0],rawidxshape[1],4,3])
    all_centcrd = np.concatenate(node_crd, 0).tolist()

    def calcfunc(x):
        centralaa, transMat, rotMat = rottrans(x[0])
        centCart = np.pad(x[1], ((0, 0), (0, 0), (0, 1)), "constant", constant_values=(0, 1)).dot(transMat)[:, :,
                   :3].dot(rotMat)
        angle = centCart[:, 1, :]
        sphere3d = np.array(list(map(lambda x: quaternion3d(centralaa, x), centCart)))
        return angle, sphere3d

    anglesphere = np.array(list(map(calcfunc, zip(np.array(all_centcrd), np.array(all_knncrd)))))
    angle = anglesphere[:,0,:,:]
    sphere3d = anglesphere[:,1,:,:]

    if rawidxshape[1] != k_nn:
        angle = np.pad(angle, ((0, 0), (0, k_nn - rawidxshape[1]), (0, 0)), 'constant', constant_values=(0, 100))
        sphere3d = np.pad(sphere3d, ((0, 0), (0, k_nn - rawidxshape[1]), (0, 0)), 'constant', constant_values=(0, 0))

    return angle.tolist(), sphere3d.tolist()


def calc_cif_interfacedistance(cif_dict, my_chain, breakchains, k_nn=20, datatype = "gz", otherchain = True):

    # if datatype == "gz":
    #     dict = GZMMCIF2Dict(cif)
    # elif datatype == "mmCIF":
    #     from Bio.PDB.MMCIFParser import MMCIF2Dict
    #     dict = MMCIF2Dict(cif)

    dict = cif_dict

    new_dict = {"ATOM":dict["_atom_site.group_PDB"],"atom_id":dict["_atom_site.auth_seq_id"],"chain":dict["_atom_site.auth_asym_id"],
                "AA_type":dict["_atom_site.label_comp_id"], "icode":dict["_atom_site.pdbx_PDB_ins_code"],
                "Atom_type":dict['_atom_site.label_atom_id'], "X":dict["_atom_site.Cartn_x"], "Y":dict["_atom_site.Cartn_y"], "Z":dict["_atom_site.Cartn_z"],
                "altloc":dict['_atom_site.label_alt_id'], "model_num": dict['_atom_site.pdbx_PDB_model_num']}
    df = pd.DataFrame.from_dict(new_dict)
    df = df[df["model_num"] == list(set(dict['_atom_site.pdbx_PDB_model_num']))[0]]

    mainatom = ["N", "CA", "C", "O"]

    df['AA_type'].replace(PROTEINLETTER3TO1, inplace=True)

    altloclist = list(set(df[(df["chain"] == my_chain) & (df["ATOM"] == "ATOM")]["altloc"].tolist()))

    if "." in altloclist:
        atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(mainatom)) & (df["ATOM"] == "ATOM") & (df["altloc"] == ".")]
    else:
        atomdf = df[(df["icode"] == "?") & (df["Atom_type"].isin(mainatom)) & (df["ATOM"] == "ATOM") & (df["altloc"] == altloclist[0])]

    countlist = atomdf.iloc[:,[1,2]].to_numpy().tolist()
    countlist = list(map(lambda x:"".join(x),countlist))
    atomdf = atomdf.assign(atom_idchain=pd.Series(countlist).values)


    # from collections import Counter
    count = Counter(countlist)
    count4df = pd.DataFrame.from_records(np.array(list(count.items())))
    allless4atom = count4df[count4df.iloc[:,1] != "4"].iloc[:,0].to_numpy()
    filterdf = atomdf[~atomdf["atom_idchain"].isin(allless4atom)]

    if otherchain == True:
        pass
    else:
        filterdf = filterdf[filterdf["chain"] == my_chain]

    CA_crd = filterdf[filterdf["Atom_type"] == "CA"].iloc[:,6:9].to_numpy().astype(np.float64)

    distCA = pdist(CA_crd, metric='euclidean')
    distMat = squareform(distCA)

    filterdf["atom_id"] = filterdf["atom_id"].astype(np.int32)
    df = filterdf[filterdf["Atom_type"] == "CA"].reset_index()

    df["AA_type"].replace(ENCODEAA2NUM,inplace=True)
    df["atom_id"] = df["atom_id"].astype(int)

    allresnum = df.shape[0]

    idx_list = df[(df["chain"] == my_chain) & (df["atom_id"].isin(np.concatenate(breakchains).tolist()))].index.tolist()
    divdistMat = distMat[idx_list]
    ignoreidx0 = np.argsort(divdistMat)

    if allresnum >= k_nn + 1:
        k_nnidx = ignoreidx0[:, 1: 1 + k_nn]
        PDBchainresnum = list(map(lambda x: df.iloc[x, [3, 2]].values.tolist(), k_nnidx))
        knnAA = list(map(lambda x: df.iloc[x, 4].values.tolist(), k_nnidx))

    else:
        raise ValueError("Sequence length must larger than knn, but get sequence length %d, knn %d" % (allresnum, k_nn))

    k_nntmpdist = np.sort(divdistMat)

    if allresnum >= k_nn + 1:
        k_nndist = k_nntmpdist[:, 1: 1 + k_nn]
    else:
        raise ValueError("Sequence length must larger than knn, but get sequence length %d, knn %d" % (allresnum, k_nn))

    return np.array(PDBchainresnum), np.array(k_nndist), filterdf, df, k_nnidx, knnAA, k_nn


def calc_dihedral(crd, includeside=False):
    phi = []
    psi = []
    omega = []
    for breakchain in crd:
        breakchain = np.array(breakchain)
        atom_indices = breakchain.shape[0]
        N_idx = [i * 4 for i in range(atom_indices // 4)]  # N
        CA_idx = [i * 4 + 1 for i in range(atom_indices // 4)]  # CA
        C_idx = [i * 4 + 2 for i in range(atom_indices // 4)]  # C
        O_idx = [i * 4 + 3 for i in range(atom_indices // 4)]  # O
        if includeside == False:
            phi_list = list(zip(C_idx[:-2], N_idx[1:-1], CA_idx[1:-1], C_idx[1:-1]))
            psi_list = list(zip(N_idx[1:-1], CA_idx[1:-1], C_idx[1:-1], N_idx[2:]))
            omega_list = list(zip(CA_idx[0:-2], C_idx[0:-2], N_idx[1:-1], CA_idx[1:-1]))
        else:
            phi_list = list(zip(C_idx[:-1], N_idx[1:], CA_idx[1:], C_idx[1:]))
            psi_list = list(zip(N_idx[0:-1], CA_idx[0:-1], C_idx[0:-1], N_idx[1:]))
            omega_list = list(zip(CA_idx[0:-1], C_idx[0:-1], N_idx[1:], CA_idx[1:]))

        sing_phi = []
        sing_psi = []
        sing_omega = []
        list_sing = list(range(len(phi_list)))

        for sing in list_sing:
            sing_phi.append(torsion(breakchain[list(phi_list[sing])][0],
                                    breakchain[list(phi_list[sing])][1],
                                    breakchain[list(phi_list[sing])][2],
                                    breakchain[list(phi_list[sing])][3]))
            sing_psi.append(torsion(breakchain[list(psi_list[sing])][0],
                                    breakchain[list(psi_list[sing])][1],
                                    breakchain[list(psi_list[sing])][2],
                                    breakchain[list(psi_list[sing])][3]))
            sing_omega.append(torsion(breakchain[list(omega_list[sing])][0],
                                      breakchain[list(omega_list[sing])][1],
                                      breakchain[list(omega_list[sing])][2],
                                      breakchain[list(omega_list[sing])][3]))

        if includeside == True:
            sing_phi.insert(0, 360)
            sing_psi.append(360)
            sing_omega.append(360)

        phi.append(sing_phi)
        psi.append(sing_psi)
        omega.append(sing_omega)

    # #### test rama & omega angle ####
    #     phi.extend(sing_phi)
    #     psi.extend(sing_psi)
    #     omega.extend(sing_omega)
    # import matplotlib.pyplot as plt
    # plt.scatter(phi, psi)
    # plt.show()
    # kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100)
    # plt.hist(omega, **kwargs)
    # plt.show()

    return phi, psi, omega