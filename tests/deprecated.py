"""
All operations in this package are deprecated or will be deprecated. The operations 
should only be used in unit tests to check consistency.
"""
import torch


def pack(atoms, adjs, proteins, device):
    atoms = torch.FloatTensor(atoms)
    adjs = torch.FloatTensor(adjs)
    proteins = torch.FloatTensor(proteins)
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0]+1)
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atoms_len += 1
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, 1:a_len+1, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        adjs_new[i,0,:] = 1
        adjs_new[i,:,0] = 1
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, 1:a_len+1, 1:a_len+1] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len),dtype=torch.int64, device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)