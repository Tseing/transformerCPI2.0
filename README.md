# TransfomerCPI2.0

  We only disclose the inference models. TransformerCPI2.0 is based on TransformerCPI whose codes are all released. The details of TransformerCPI2.0 are described in our paper https://doi.org/10.1038/s41467-023-39856-w which is now published on Nature communications. Trained models are available at present.

## Setup and dependencies 

`environment.yaml` is the conda environment of this project.

## Inference

`predict.py` makes the inference, the input are protein sequence and compound SMILES. `featurizer.py` tokenizes and encodes the protein sequence and compounds. `mutation_analysis.py` conducts drug mutation analysis to predict binding sites. `substitution_analysis.py` conducts substitution analysis.

## Trained models

Trained model is originally from https://drive.google.com/drive/folders/1X7i1eO-EykCQcvqMeWeB7QXT3E9eLG08?usp=sharing. But the model are incompatible with specified environment when using `torch.load()` to load a model directly.

Original model is detached into pretrained BERT (`pretrained_bert.pt`) and parameters of Predictor backbone model (`virtual_screening.pt`). The detached models are available at https://drive.google.com/drive/folders/1qu9OXRwfijUr51pXEnOZiZ_JkxjXndej.

Please load models in steps recommended by PyTorch like this:

```py
# load pretrained BERT
pre-trained_model = torch.load('pretrained_bert.pt')
# initialize backbone model
model = Predictor(pretrained_model, device=device)
# load parameters of backbone model
model.load_state_dict(torch.load('virtual_screening.pt'))
```

## Requirements

python==3.9.18

torch==1.10.0+cu111torch 

tape-proteins==0.5

rdkit==2023.9.2

numpy==1.26.2

scikit-learn==1.3.2

