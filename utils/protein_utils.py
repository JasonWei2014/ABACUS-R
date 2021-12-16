import numpy as np

ENCODESS82NUM = {
    "H": 0,
    "B": 1,
    "E": 2,
    "G": 3,
    "I": 4,
    "T": 5,
    "S": 6,
    "-": 7
}

ENCODESS32NUM = {
    "H": 0,
    "L": 1,
    "E": 2
}

ENCODEAA2NUM = {
    "X": -1,
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

PROTEINLETTER3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

#### {"ASX": "ASP", "XAA": "GLY", "GLX": "GLU", "XLE": "LEU", "SEC": "CYS", "PYL": "LYS", "UNK": "GLY"}
non_standardAA = {"ASX": "D", "XAA": "G", "GLX": "E", "XLE": "L", "SEC": "C", "PYL": "K", "UNK": "G", "PTR": "Y"}
PROTEINLETTER3TO1.update(non_standardAA)

RESIDUEMAXACC = {
    # Miller max acc: Miller et al. 1987 https://doi.org/10.1016/0022-2836(87)90038-6
    # Wilke: Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635
    # Sander: Sander & Rost 1994 https://doi.org/10.1002/prot.340200303
    "Miller": {
        "ALA": 113.0,
        "ARG": 241.0,
        "ASN": 158.0,
        "ASP": 151.0,
        "CYS": 140.0,
        "GLN": 189.0,
        "GLU": 183.0,
        "GLY": 85.0,
        "HIS": 194.0,
        "ILE": 182.0,
        "LEU": 180.0,
        "LYS": 211.0,
        "MET": 204.0,
        "PHE": 218.0,
        "PRO": 143.0,
        "SER": 122.0,
        "THR": 146.0,
        "TRP": 259.0,
        "TYR": 229.0,
        "VAL": 160.0,
    },
    "Wilke": {
        "ALA": 129.0,
        "ARG": 274.0,
        "ASN": 195.0,
        "ASP": 193.0,
        "CYS": 167.0,
        "GLN": 225.0,
        "GLU": 223.0,
        "GLY": 104.0,
        "HIS": 224.0,
        "ILE": 197.0,
        "LEU": 201.0,
        "LYS": 236.0,
        "MET": 224.0,
        "PHE": 240.0,
        "PRO": 159.0,
        "SER": 155.0,
        "THR": 172.0,
        "TRP": 285.0,
        "TYR": 263.0,
        "VAL": 174.0,
    },
    "Sander": {
        "ALA": 106.0,
        "ARG": 248.0,
        "ASN": 157.0,
        "ASP": 163.0,
        "CYS": 135.0,
        "GLN": 198.0,
        "GLU": 194.0,
        "GLY": 84.0,
        "HIS": 184.0,
        "ILE": 169.0,
        "LEU": 164.0,
        "LYS": 205.0,
        "MET": 188.0,
        "PHE": 197.0,
        "PRO": 136.0,
        "SER": 130.0,
        "THR": 142.0,
        "TRP": 227.0,
        "TYR": 222.0,
        "VAL": 142.0,
    },
}

def transform_aa3to1(AAaray):
    return np.array([list(map(lambda x: PROTEINLETTER3TO1[x], AAaray[i])) for i in range(AAaray.shape[0])])
