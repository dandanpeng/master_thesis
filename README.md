# Deep Generative Models for the Computational Design of T-cell Receptor CDR3 sequences
This repository includes several Variational Autoencoder variants to fit the probability distribution and generate for T cell receptor beta chain CDR3 sequences.

## Setting up environment
Our work is inspired by [Vampire](https://github.com/matsengrp/vampire). Before running our VAE models, you should set up your environment as described in Vampire repository.

## Preprocessing Data
- Amino acid sequences are onehot-encoded into 30*21 matrix. Processing step can be finished by executing `preprocess_adaptive.py` in Vampire.
- Nucleotide sequence are onehot-encoded into 90*5 matrix. Processing step can be finished by executing `preprocess_nt.py`.

## Processed Data
The data we used to train the models are cohort2 from [Immunosequencing identifies signatures of cytomegalovirus exposure history and HLA-mediated effects on the T-cell repertoire](https://clients.adaptivebiotech.com/pub/5dd7b508-079b-4cf6-872d-4a91e5e3e5db).
Processed data can be downloaded from [Zenodo](https://zenodo.org/record/3931962#.XwNGk5Mzblw).

## Running
Before running `.py` scripts, you should execute
```
conda activate vampire
```
### VAE-seq (vae_seq.py)
- Input and output are CDR3 amino acid sequence.
- Encoder and decoder consists dense layers.

### Supervised VAE (svae.py)
- Input data is CDR3 amino acid sequence and the corresponding V- and J- genes.
- Output data is CDR3 amino acid sequence.
- Encoder and decoder consists dense layers.

### Nuclotide VAE (nt_vae.py)
- Input data is CDR3 nucleotide sequence.
- Output data is CDR3 amino acid sequence.
- Encoder and decoder consists dense layers.

### Recurrent neural network VAE (rnn_vae.py)
- Input and output are CDR3 amino acid sequence.
- Encode consists of bidirectional GRU layer, decoder consists of undirectional GRU layer.

## Result analysis
The scripts that plot distance heatmap of generations' probability distribution and scatter of frequency estimation are `modeller_plot.py` and `plot_heatmap.py`.
