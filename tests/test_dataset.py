import torch

import tests.deprecated as deprecated
from featurizer import featurizer
from predict import pack
from tests.utils import check_shape_value


class TestDataProcess:
    def setup_class(self):
        self.device = torch.device("cuda:0")
        sequence = ["MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA"]
        smiles = ["CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O"]
        self.compounds, self.adjacencies, self.proteins = featurizer(smiles, sequence)

    def test_pack(self):
        expected_tuple = deprecated.pack(
            self.compounds, self.adjacencies, self.proteins, self.device
        )

        result_tuple = pack(self.compounds, self.adjacencies, self.proteins, self.device)
        
        vars = ["padded_atoms", "padded_adjs", "padded_proteins", "mols_len", "proteins_len"]
        for i, pair in enumerate(zip(expected_tuple, result_tuple)):
            expected, result = pair
            check_shape_value(expected, result, vars[i])
