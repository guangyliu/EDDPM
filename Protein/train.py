import datetime
import time
import os
import numpy as np
import argparse
from argparse import ArgumentParser
from sklearn.decomposition import PCA

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint_adjoint

import torch
import torch.nn as nn
import torch.nn.init as init

from codes.nn.models import relso1, DDPM, RelsoDiffusion, MLPSkipNet, RelsoVAE
import codes.data as hdata
from codes.utils import eval_utils
from codes.optim import optim_algs, utils

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

from evaluations import SEQ2IND, IND2SEQ
from tqdm import tqdm 
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import wandb


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)

if __name__ == "__main__":

    tic = time.perf_counter()

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument('--n_steps', default=200, type=int)

    # data argmuments
    parser.add_argument("--input_dim", default=22, type=int)
    parser.add_argument("--task", default="recon", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--project_name", default="JEDI_Protein_Final", type=str)
    parser.add_argument("--cpu", default=False, action="store_true")

    # training arguments
    parser.add_argument("--alpha_val", default=1.0, type=float)
    parser.add_argument("--beta_val", default=0.0005, type=float)
    parser.add_argument("--gamma_val", default=1.0, type=float)
    parser.add_argument("--sigma_val", default=1.5, type=float)

    parser.add_argument("--eta_val", default=0.001, type=float)

    parser.add_argument("--reg_ramp", default=False, type=str2bool)
    parser.add_argument("--vae_ramp", default=True, type=str2bool)

    parser.add_argument("--neg_samp", default=False, type=str2bool)
    parser.add_argument("--neg_size", default=16, type=int)
    parser.add_argument("--neg_weight", default=0.8, type=float)
    parser.add_argument("--neg_floor", default=-2.0, type=float)
    parser.add_argument("--neg_norm", default=4.0, type=float)
    parser.add_argument("--neg_focus", default=False, type=str2bool)

    parser.add_argument("--interp_samp", default=False, type=str2bool)
    parser.add_argument("--interp_size", default=16, type=int)
    parser.add_argument("--interp_weight", default=0.001, type=float)

    parser.add_argument("--wl2norm", default=False, type=str2bool)

    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--n_epochs", default=300, type=int)
    parser.add_argument("--n_gpus", default=0, type=int)
    parser.add_argument("--dev", default=False, type=str2bool)
    parser.add_argument("--seq_len", default=0, type=int)
    parser.add_argument("--auto_lr", default=False, type=str2bool)
    parser.add_argument("--seqdist_cutoff", default=None)

    # LSTM
    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)

    # CNN
    parser.add_argument("--kernel_size", default=4, type=int)
    
    # DDPM
    parser.add_argument('--ddpm_weight', type=float, default=1.0)
    parser.add_argument('--nt', type=int, default=1000, help="T for diffusion process")
    parser.add_argument('--joint_regressor', action='store_true')

    # BOTH
    parser.add_argument("--latent_dim", default=30, type=int)
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--probs", default=0.2, type=float)
    parser.add_argument("--auxnetwork", default="base_reg", type=str)
    parser.add_argument('--only_vae', action='store_true')
    parser.add_argument("--kl_weight", default=1.0, type=float)

    # Optimization Parameters
    parser.add_argument('--y_label', type=float, default=1.0)
    parser.add_argument('--split', type=str, default="train")

    parser.add_argument('--det_inits', default=False, action='store_true')
    parser.add_argument('--alpha', required=False, type=float)
    parser.add_argument('--delta', required=False, default='adaptive', type=str)
    parser.add_argument('--k', required=False, default=5, type=float)

    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    print("now training")
    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # ---------------------------
    # LOGGING
    # ---------------------------
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")

    if cl_args.joint_regressor:
        if cl_args.only_vae:
            save_dir = f"logs/{cl_args.dataset}/vae_regressor_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_kl{cl_args.kl_weight}/{date_suffix}/"
        else:
            save_dir = f"logs/{cl_args.dataset}/relso_diffusion_regressor_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}/{date_suffix}/"
    else:
        if cl_args.only_vae:
            save_dir = f"logs/{cl_args.dataset}/vae_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_kl{cl_args.kl_weight}/{date_suffix}/"
        else:
            save_dir = f"logs/{cl_args.dataset}/relso_diffusion_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}/{date_suffix}/"

    if cl_args.joint_regressor:
        if cl_args.only_vae:
            logger_name = f"vae_regressor_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_{cl_args.dataset}_kl{cl_args.kl_weight}_{date_suffix}"
        else:
            logger_name = f"diffusion_regressor_relso_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_{cl_args.dataset}_{date_suffix}"
    else:
        if cl_args.only_vae:
            logger_name = f"vae_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_{cl_args.dataset}_kl{cl_args.kl_weight}_{date_suffix}"
        else:
            logger_name = f"diffusion_relso_w{cl_args.ddpm_weight}_dim{cl_args.latent_dim}_{cl_args.dataset}_{date_suffix}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb_logger = WandbLogger(
        name=logger_name,
        project=cl_args.project_name,
        log_model=True,
        save_dir=save_dir,
        offline=False,
        entity='tedfeng424',
    )

    wandb_logger.log_hyperparams(cl_args.__dict__)
    wandb_logger.experiment.log({"logging timestamp": date_suffix})

    # ---------------------------
    # TRAINING
    # ---------------------------
    early_stop_callback = EarlyStopping(
        monitor="valid fit smooth",  # set in EvalResult
        min_delta=0.001,
        patience=8,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,  # Directory where checkpoints are saved
        filename='{epoch}',     # Checkpoint file name
        every_n_epochs=50,      # Save a checkpoint every 1 epochs
        save_top_k=-1,          # Set to -1 to save all checkpoints
        verbose=True            # Set to True to get log messages
    )

    # get models and data
    proto_data = hdata.str2data(cl_args.dataset)

    # initialize both model and data
    data = proto_data(
        dataset=cl_args.dataset,
        task=cl_args.task,
        batch_size=cl_args.batch_size,
        seqdist_cutoff=cl_args.seqdist_cutoff,
    )

    cl_args.seq_len = data.seq_len

    # ODE Hyperparameter
    sde_kwargs = {'N': 1000, 'correct_nsteps': 2, 'target_snr': 0.16}
    ode_kwargs = {'atol': 1e-3, 'rtol': 1e-3, 'method': 'dopri5', 'use_adjoint': True, 'latent_dim': cl_args.latent_dim}
    ld_kwargs = {'batch_size': 200, 'sgld_lr': 1,
                 'sgld_std': 1e-2, 'n_steps': 200}

    if cl_args.only_vae:
        model = RelsoVAE(hparams=cl_args)
    else:
        model = RelsoDiffusion(hparams=cl_args)
        
    if cl_args.joint_regressor is False:
        for p in model.regressor_module.parameters():
            p.requires_grad_(False)
        print("No regressor")

    if cl_args.cpu:
        trainer = pl.Trainer.from_argparse_args(
            cl_args,
            max_epochs=cl_args.n_epochs,
            accelerator="cpu",
            # devices=1,
            log_every_n_steps=min(len(data.train_dataloader()) // 2, 50),
            #logger=wandb_logger,

        )

    else:
        cl_args.n_gpus = 1
        trainer = pl.Trainer.from_argparse_args(
            cl_args,
            callbacks=[checkpoint_callback],
            max_epochs=cl_args.n_epochs,
            # strategy="dp",
            accelerator="gpu",
            devices=cl_args.n_gpus,
            log_every_n_steps=min(len(data.train_dataloader()) // 2, 50),
            logger=wandb_logger,
            check_val_every_n_epoch=1,
        )

    # Run learning rate finder if selected
    if cl_args.auto_lr:
        print("auto learning rate enabled")
        print("selecting optimal learning rate")
        lr_finder = trainer.tuner.lr_find(
            model, train_dataloader=data.train_dataloader()
        )

        # pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("old lr: {} | new lr: {}".format(cl_args.lr, new_lr))

        # update hparams of the model
        model.hparams.lr = new_lr
        wandb_logger.experiment.log({"auto_find_lr": new_lr})
    
    import time
    start = time.time()
    trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.valid_dataloader(),
    )

    # save model
    trainer.save_checkpoint(save_dir + "model_state.ckpt")

