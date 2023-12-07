# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/10/13 13:54
@author: LiFan Chen
@Filename: predict.py
@Software: PyCharm
"""

import numpy as np
import torch
from numpy import ndarray

from featurizer import featurizer
from model import Predictor


class Tester(object):
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def test(self, dataset: tuple) -> ndarray:
        self.model.eval()
        with torch.no_grad():
            atoms, adjs, proteins = dataset[0], dataset[1], dataset[2]
            data = pack(atoms, adjs, proteins, self.device)
            predicted_scores = self.model(data)
        return predicted_scores


def pack(mols: ndarray, adjs: ndarray, proteins: ndarray, device: torch.device):
    def pad_mol(mol: ndarray) -> ndarray:
        return np.pad(
            mol, ((1, mols_max_len - mol.shape[0] - 1), (0, 0)), "constant", constant_values=0
        )

    def pad_adj(adj: ndarray) -> ndarray:
        padded_adj = np.pad(
            adj + np.eye(adj.shape[0]), (0, mols_max_len - adj.shape[0] - 1), constant_values=0
        )
        padded_adj = np.pad(padded_adj, (1, 0), "constant", constant_values=1)
        return padded_adj

    def pad_protein(protein: ndarray) -> ndarray:
        return np.pad(protein, (0, proteins_max_len - protein.shape[0]), constant_values=0)

    bsz = len(mols)
    assert bsz == len(adjs) and bsz == len(proteins), "Batch sizes of data are different."

    # mols_len keep atoms number of per molecule
    # mols will be padded, add 1 to original length
    mols_len = np.array(list(map(lambda x: x.shape[0], mols))) + 1
    mols_max_len = max(mols_len)

    proteins_len = np.array(list(map(lambda x: x.shape[0], proteins)))
    proteins_max_len = max(proteins_len)

    padded_atoms = torch.tensor(
        np.array([pad_mol(mol) for mol in mols]), dtype=torch.float32, device=device
    )
    padded_adjs = torch.tensor(
        np.array([pad_adj(adj) for adj in adjs]), dtype=torch.float32, device=device
    )
    padded_proteins = torch.tensor(
        np.array([pad_protein(protein) for protein in proteins]), dtype=torch.int64, device=device
    )

    return (padded_atoms, padded_adjs, padded_proteins, mols_len, proteins_len)


if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("The code uses GPU...")
    else:
        device = torch.device("cpu")
        print("The code uses CPU!!!")

    pretrained_model = torch.load("pretrained_bert.pt")
    model = Predictor(pretrained_model, device=device)
    model.load_state_dict(torch.load("virtual_screening.pt"))
    model.to(device)
    # Example protein sequence
    sequence = [
        "MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA",
        "MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHT",
    ]
    # Example compound
    smiles = ["CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O", "Cc1ccc(O)c2c1C1CCCN(C)C1CO2"]
    # (bsz, atoms_num, 34), (bsz, atoms_num, atoms_num), (bsz, 54)
    compounds, adjacencies, proteins = featurizer(smiles, sequence)
    tester = Tester(model, device)
    score = tester.test((compounds, adjacencies, proteins))
    print(score)
