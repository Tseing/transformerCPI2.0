# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/10/13 13:54
@author: LiFan Chen
@Filename: predict.py
@Software: PyCharm
"""
import sys

sys.path.append("..")

import torch

from src.model.transformer import Predictor, Tester
from src.model.utils import featurizer

if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("The code uses GPU...")
    else:
        device = torch.device("cpu")
        print("The code uses CPU!!!")

    pretrained_model = torch.load("../pretrained_bert.pt")
    model = Predictor(pretrained_model, device=device)
    model.load_state_dict(torch.load("../virtual_screening.pt"))
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
