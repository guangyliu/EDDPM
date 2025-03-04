This the implementation of the protein sequence experiments of the paper **Generating, Reconstructing, and Representing Discrete and Continuous Data: Generalized Encoding-Decoding Diffusion Probabilistic Models**. 

This code base is build upon [ReLSO](https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers/tree/main), a protein sequence Autoencoder.

The dataset for saved under `data` folder.

## Training
To run the training:
```
python train.py --dataset [DATASET] --joint_regressor --latent_dim [LATENT_DIM]
```
`--dataset` specifies the which dataset to use, can selected from {GFP, gifford}. 


`--joint_regressor` specifies whether to jointly train a regressor that predicts the fitness value of the protein sequences.

`--latent_dim` specifies the dimension of the latent space(we used 30 to be consistent with experiments ran by ReLSO).

`--ddpm_weight` specifies the weight multiplied to diffusion loss(we used the default value of 1.0).

## Evaluation
To run the evaluation:
```
python evaluations.py --dataset [DATASET] --split [SPLIT] --model_path [MODELS] --save_path [SAVE_PATH]
```
`--dataset` specifies which dataset to use, can select from {GFP, gifford}.

`--split` specifies which split to run evaluations on, can select from {train, valid, test}.

`--model_path` specifies the path to model checkpoints, this path can contain multiple checkpoints and the codes will run evaluations on all checkpoints.

`save_path` specifies the path to save the evaluation results.


