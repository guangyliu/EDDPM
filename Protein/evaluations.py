import datetime
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

from tqdm import tqdm 
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import wandb
from natsort import natsorted

SEQ2IND = {"I":0,
          "L":1,
          "V":2,
          "F":3,
          "M":4,
          "C":5,
          "A":6,
          "G":7,
          "P":8,
          "T":9,
          "S":10,
          "Y":11,
          "W":12,
          "Q":13,
          "N":14,
          "H":15,
          "E":16,
          "D":17,
          "K":18,
          "R":19,
          "X":20,
          "J":21}

IND2SEQ = {ind: AA for AA, ind in SEQ2IND.items()}


def get_evaluation(embedding_path, model, model_ppl, dataset_name):

    from tqdm import tqdm

    *_, train_targs = data.train_split.tensors
    train_targs = train_targs.numpy()

    # Get Embeddings
    train_reps, train_sequences, train_targs = data.train_split.tensors

    if 'unconditional' in embedding_path:
        embeddings = np.load(embedding_path, allow_pickle=True).astype(np.float32)
    else:
        embeddings = np.load(embedding_path, allow_pickle=True)

    if 'unconditional' in embedding_path:
        optim_algo_names = ['Generation']
    else:
        if dataset_name == 'gifford':
            optim_algo_names = ['MCMC', 'MCMC-cycle', 'MCMC-cycle-noP','Hill Climbing', 'Stochastic Hill Climbing','Gradient Ascent', 'LatentOps1', 'LatentOps15', 'LatentOps2', 'LatentOps25']
        else:
            optim_algo_names = ['MCMC', 'MCMC-cycle', 'MCMC-cycle-noP','Hill Climbing', 'Stochastic Hill Climbing','Gradient Ascent', 'LatentOps3', 'LatentOps4']

    df = pd.DataFrame(columns=['model_name', 'max_fitness', 'mean_fitness', 'median_fitness', 'min_fitness', 'diversity', 'novelty'])
    df['model_name'] = optim_algo_names

    fitnesses = {}
    for item in optim_algo_names:
        fitnesses[item] = []
    seqs = {}
    for item in optim_algo_names:
        seqs[item] = []

    if 'unconditional' in embedding_path:
        for emb in tqdm(embeddings):
            embed_dim = emb.shape[-1]
            try:
                curr_embedding = emb.reshape(1,embed_dim)
            except: 
                embed()
            fitness = optim_algs.eval_oracle(curr_embedding, model.regressor_module)
            curr_embedding = torch.from_numpy(curr_embedding).float().reshape(1,-1).to(model.device)
            with torch.no_grad():
                decoded_seq = model.decode(curr_embedding).argmax(1)
            seq = decoded_seq.cpu().numpy()[0]
            seq = " ".join([IND2SEQ[l] for l in seq])

            fitnesses[optim_algo_names[0]].append(fitness)
            seqs[optim_algo_names[0]].append(seq)
    else:
        for runs in tqdm(embeddings):
            for i in range(len(runs)):
                if i >= 6:
                    algo_embedding = runs[i]
                else:
                    algo_embedding = runs[i][-1]
                embed_dim = algo_embedding.shape[-1]
                try:
                    curr_embedding = algo_embedding.reshape(1,embed_dim)
                except: 
                    embed()
                fitness = optim_algs.eval_oracle(curr_embedding, model.regressor_module)
                curr_embedding = torch.from_numpy(curr_embedding).float().reshape(1,-1).to('cuda')
                with torch.no_grad():
                    decoded_seq = model.decode(curr_embedding).argmax(1)
                seq = decoded_seq.cpu().numpy()[0]
                seq = " ".join([IND2SEQ[l] for l in seq])

                fitnesses[optim_algo_names[i]].append(fitness)
                seqs[optim_algo_names[i]].append(seq)

    max_fitness = []
    mean_fitness = []   
    median_fitness = []
    min_fitness = [] 
    diversity = []
    novelty = []

    for algo in tqdm(optim_algo_names):
        max_fitness.append(np.max(fitnesses[algo]))
        mean_fitness.append(np.mean(fitnesses[algo]))
        median_fitness.append(np.median(fitnesses[algo]))
        min_fitness.append(np.min(fitnesses[algo]))

        algo_sequences = seqs[algo]
        lev_distances = []
        for i, sequence in enumerate(algo_sequences):
            new_seqs = np.delete(algo_sequences, i, 0)
            for new_seq in new_seqs:
                ld = levenshtein_distance(sequence, new_seq)
                lev_distances.append(ld)
        diversity.append(np.mean(lev_distances))

        single_novelty = []
        for i, sequence in enumerate(algo_sequences):
            
            train_ld = []
            for train_seq in train_sequences:

                train_seq = " ".join([IND2SEQ[l] for l in train_seq.numpy()])
                distance_nov = levenshtein_distance(sequence, train_seq)
                train_ld.append(distance_nov)
            
            single_novelty.append(np.min(train_ld))
            
        novelty.append(np.median(single_novelty))

    #print("Load ppl model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculatePerplexity(sequence, model, tokenizer):
        input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        return math.exp(loss), loss.item()

    ppl_dict = []
    nll_dict = []
    for algo in tqdm(optim_algo_names):

        ppls = []
        nlls = []
        algo_sequences = seqs[algo]
        for i, sequence in enumerate(algo_sequences):
            seq_input = "<|endoftext|>"+sequence+"<|endoftext|>"
            seq_ppl, seq_nll = calculatePerplexity(seq_input, model_ppl, tokenizer)
            ppls.append(seq_ppl)
            nlls.append(seq_nll)

        ppl_dict.append(np.mean(ppls))
        nll_dict.append(np.mean(nlls))


    df['max_fitness'] =  max_fitness
    df['mean_fitness'] =  mean_fitness
    df['median_fitness'] =  median_fitness
    df['min_fitness'] =  min_fitness
    df['diversity'] = diversity
    df['novelty'] = novelty
    df['ppl'] = ppl_dict
    df['nll'] = nll_dict

    return df

class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None, every_n_plot=5,
                kwargs=None):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y
        self.save_path = save_path
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.plot = plot

    def forward(self, t_k, states):
        z = states[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime
        self.n_evals += 1
        return dz_dt,

class CCF(nn.Module):
    def __init__(self, classifier, cl_args):
        super(CCF, self).__init__()
        self.f = nn.ModuleList()
        for cls in classifier:
            self.f.append(cls)

    def get_cond_energy(self, z, y_):
        energy_outs = []
        for i in range(y_.shape[1]):
            cls = self.f[i]
            logits = cls(z)
            n_classes = logits.size(1)
            energy_weight = cl_args.energy_weight
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  
                sigle_energy = -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
                energy_outs.append(energy_weight * sigle_energy)
                
        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output 
    def get_cond_energy_single(self, z, y_):
        for i in range(y_.shape[1]):
            energy_outs = []
            cls = self.f[i]
            logits = cls(z)
            n_classes = logits.size(1)
            energy_weight = cl_args.energy_weight
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  # this value works well
                sigle_energy = -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
                energy_outs.append(energy_weight * sigle_energy)
               
        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.norm(z, dim=1) ** 2 * 0.5
        return energy_output


def sample_q_ode(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']
    kwargs['device'] = device
    z_k = kwargs['z_k']
    # ODE function
    vpode = VPODE(ccf, y, save_path=save_path, plot=plot, every_n_plot=every_n_plot, kwargs=kwargs)
    states = (z_k,)
    integration_times = torch.linspace(vpode.T, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,  # model
        states,  # (z,)
        integration_times,
        atol=atol,  # tolerance
        rtol=rtol,
        method=method)

    ccf.train()
    z_t0 = state_t[0][-1]
    return z_t0.detach(), vpode.n_evals

def get_fitness(init_point, embedding, oracle):
    embed_dim = init_point.shape[-1]
    curr_embedding = embedding.reshape(1,embed_dim)
    curr_fit = optim_algs.eval_oracle(curr_embedding, oracle)

    return curr_fit


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
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)

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
    parser.add_argument("--kl_weight", default=1.0, type=float)

    # Optimization Parameters
    parser.add_argument('--y_label', type=float, default=1.0)
    parser.add_argument('--energy_weight', type=float, default=1.0)
    parser.add_argument('--split', type=str, default="train")

    parser.add_argument('--det_inits', default=False, action='store_true')
    parser.add_argument('--alpha', required=False, type=float)
    parser.add_argument('--delta', required=False, default='adaptive', type=str)
    parser.add_argument('--k', required=False, default=5, type=float)

    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--save_path', required=True, type=str)

    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # ---------------------------
    # LOGGING
    # ---------------------------

    logger_name = 'eval_test'
    logger_save_dir = f"{cl_args.model_path}_evaluations"
    if not os.path.exists(logger_save_dir):
        os.makedirs(logger_save_dir)

    wandb_logger = WandbLogger(
        name=logger_name,
        project=cl_args.project_name,
        log_model=True,
        save_dir=logger_save_dir,
        offline=False,
        entity='tedfeng424',
    )


    # get data
    proto_data = hdata.str2data(cl_args.dataset)

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

    ckpts = os.listdir(cl_args.model_path)
    ckpts = natsorted([os.path.join(cl_args.model_path, ck) for ck in ckpts if 'epoch' in ck and 'ckpt' in ck])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_eval_results = []

    model_path2 = "/data/ted/models/ProtGPT2_ppl"
    tokenizer = AutoTokenizer.from_pretrained(model_path2)
    model_ppl = AutoModelForCausalLM.from_pretrained(model_path2).to(device)

    for checkpoint in ckpts:
        model_name = f"{checkpoint.split('/')[7]}_{checkpoint.split('/')[-1].split('.')[0]}"
        save_dir = os.path.join(cl_args.save_path, f"eval_{checkpoint.split('/')[-1].split('.')[0] }")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if 'vae' in cl_args.model_path:
            model = RelsoVAE.load_from_checkpoint(checkpoint)
        else:
            model = RelsoDiffusion.load_from_checkpoint(checkpoint)

        model.eval()
        model.cpu()

        # ---------------------
        # EVALUATION
        # ---------------------
        # Load raw data using load_rawdata, which gives indices + enrichment
        train_reps, _, train_targs = data.train_split.tensors  # subset objects
        valid_reps, _, valid_targs = data.valid_split.tensors  # subset objects
        test_reps, _, test_targs = data.test_split.tensors

        print("train sequences raw shape: {}".format(train_reps.shape))
        print("valid sequences raw shape: {}".format(valid_reps.shape))
        print("test sequences raw shape: {}".format(test_reps.shape))

        train_n = train_reps.shape[0]
        valid_n = valid_reps.shape[0]
        test_n = test_reps.shape[0]

        print("getting embeddings")
        train_outputs, train_hrep, train_yhat = eval_utils.get_diff_model_outputs(model, train_reps, device)
        valid_outputs, valid_hrep, valid_yhat = eval_utils.get_diff_model_outputs(model, valid_reps,device)
        test_outputs, test_hrep, test_yhat = eval_utils.get_diff_model_outputs(model, test_reps, device)

        # print("model has fitness predictions")
        train_recon = train_outputs
        valid_recon = valid_outputs
        test_recon = test_outputs

        targets_list = [train_targs, valid_targs, test_targs]
        recon_targ_list = [train_reps, valid_reps, test_reps]

        predictions_list = [train_yhat, valid_yhat, test_yhat]
        recon_list = [x for x in [train_outputs, valid_outputs, test_outputs]]

        # ------------------------------------------------
        # EMBEDDING EVALUATION
        # ------------------------------------------------
        print("saving embeddings")

        train_embed = train_hrep.reshape(train_n, -1).numpy()
        valid_embed = valid_hrep.reshape(valid_n, -1).numpy()
        test_embed = test_hrep.reshape(test_n, -1).numpy()

        embed_list = [train_embed, valid_embed, test_embed]

        # ------------------------------------------------
        # FITNESS PREDICTION EVALUATIONS
        # ------------------------------------------------

        eval_results = {"model_name":model_name}

        fitness_eval_results = eval_utils.get_all_fitness_pred_metrics(
            targets_list=targets_list,
            predictions_list=predictions_list,
            wandb_logger=wandb_logger,
        )
        eval_results.update(fitness_eval_results)

        # # ------------------------------------------------
        # # RECONSTRUCTION EVALUATIONS
        # # ------------------------------------------------
        reconstruction_eval_results = eval_utils.get_all_recon_pred_metrics(
            targets_list=recon_targ_list,
            predictions_list=recon_list,
            wandb_logger=wandb_logger,
        )
        eval_results.update(reconstruction_eval_results)

        all_eval_results.append(eval_results)

        # ------------------------------------------------
        # Optimization
        # ------------------------------------------------

        cl_args.model_paths = None

        # Get Embeddings
        if cl_args.split == 'train':
            train_reps, _, train_targs = data.train_split.tensors 
        elif cl_args.split == 'valid':
            train_reps, _, train_targs = data.valid_split.tensors 
        elif cl_args.split == 'test':
            train_reps, _, train_targs = data.test_split.tensors 
        else:
            raise Exception("Given Split is not in the dataset")

        train_n = train_reps.shape[0]
        train_outputs, train_hrep, train_yhat = eval_utils.get_diff_model_outputs(model, train_reps, device)
        train_embed = train_hrep.reshape(train_n, -1).numpy()
        embeddings = train_embed

        n_steps = cl_args.n_steps
        num_inits = 60 
        if cl_args.dataset == 'gifford':
            optim_algo_names = ['MCMC', 'MCMC-cycle', 'MCMC-cycle-noP','Hill Climbing', 'Stochastic Hill Climbing','Gradient Ascent', 'LatentOps1', 'LatentOps1.5', 'LatentOps2', 'LatentOps2.5']
        else:
            optim_algo_names = ['MCMC', 'MCMC-cycle', 'MCMC-cycle-noP','Hill Climbing', 'Stochastic Hill Climbing','Gradient Ascent', 'LatentOps3', 'LatentOps4']
        num_optim_algs = len(optim_algo_names)

        optim_embedding_traj_array = np.zeros((num_inits, num_optim_algs, n_steps,  embeddings.shape[-1]))
        optim_fitness_traj_array = np.zeros((num_inits, num_optim_algs, n_steps))

        if cl_args.det_inits:
            print('deterministic seeds selected!')
            seed_vals = np.linspace(0,len(embeddings)-1, num_inits)
        else:
            print('random seeds selected!')
            seed_vals = np.random.choice(np.arange(len(embeddings)), num_inits)


        if cl_args.delta == 'adaptive':
            print('adaptive delta selected - computing delta based off pairwise distances')
            cl_args.delta = eval_utils.get_avg_distance(embeddings=embeddings, k=cl_args.k)

        else:
            cl_args.delta = float(cl_args.delta)
        
        final_embeddings = []

        for run_indx, init_indx in enumerate(seed_vals):

            init_indx = int(init_indx)

            #print(f'\nrunning initialization {run_indx}/{num_inits}\n')

            init_point = embeddings[init_indx].copy()

            # MCMC
            #print("\n")
            embedding_array_mcmc, fitness_array_mcmc = optim_algs.metropolisMCMC_embedding(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                        delta=cl_args.delta,
                                                                                        N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_mcmc.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc[0][:10]))

            # print("\n")
            embedding_array_mcmc_cycle, fitness_array_mcmc_cycle = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                                oracle=model.regressor_module,
                                                                                                model=model,
                                                                                                delta=cl_args.delta,
                                                                                                N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_mcmc_cycle.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc_cycle[0][:10]))


            # print("\n")
            embedding_array_mcmc_cycle_noP, fitness_array_mcmc_cycle_noP = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                                oracle=model.regressor_module,
                                                                                                model=model,
                                                                                                N_steps=n_steps,
                                                                                                delta=cl_args.delta,
                                                                                                perturbation=False)
            # print(f'shape of output embedding array: {embedding_array_mcmc_cycle_noP.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc_cycle_noP[0][:10]))


            # # Hill climbing
            # print("\n")
            embedding_array_hill, fitness_array_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                            dataset_embeddings=embeddings,
                                                                                            N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_hill.shape}')
            # print('init embed from output: {}'.format(embedding_array_hill[0][:10]))

            # # Stochastic climbing
            # print("\n")
            embedding_array_s_hill, fitness_array_s_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding = init_point.copy(),
                                                                                                oracle = model.regressor_module,
                                                                                                dataset_embeddings=embeddings,
                                                                                                N_steps=n_steps,
                                                                                                stochastic=True)
            # print(f'shape of output embedding array: {embedding_array_s_hill.shape}')
            # print('init embed from output: {}'.format(embedding_array_s_hill[0][:10]))

            # # Gradient Ascent
            # print("\n")
            embedding_array_ga, fitness_array_ga = optim_algs.grad_ascent(initial_embedding = init_point.copy(),
                                                                        model=model,
                                                                        N_steps=n_steps,
                                                                        lr=0.1)

            # print(f'shape of output embedding array: {embedding_array_ga.shape}')
            # print('init embed from output: {}'.format(embedding_array_ga[0][:10]))


            ########################
            # LatentOps Optimization
            ########################

            initial_embedding = init_point.copy()   
            curr_fit = get_fitness(init_point, initial_embedding,  model.regressor_module)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            classifier_list = [model.regressor_module]
            classifier = CCF(classifier_list, cl_args)
            classifier.to(device)
            y = torch.Tensor([[cl_args.y_label]]).to(device)
            init_point_tensor = torch.from_numpy(init_point).to(device).unsqueeze(dim=0)
            init_point_tensor = init_point_tensor.repeat(2, 1)

            z = init_point_tensor

            if cl_args.dataset == 'gifford':
                y = torch.Tensor([[1.0]]).to(device)
                embedding_array_latentops1  = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit1 = get_fitness(init_point, embedding_array_latentops1[0][0],  model.regressor_module)
                fitness_array_latentops1 = [curr_fit1] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[1.5]]).to(device)
                embedding_array_latentops15 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit15 = get_fitness(init_point, embedding_array_latentops15[0][0],  model.regressor_module)
                fitness_array_latentops15 = [curr_fit15] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[2.0]]).to(device)
                embedding_array_latentops2 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit2 = get_fitness(init_point, embedding_array_latentops2[0][0],  model.regressor_module)
                fitness_array_latentops2 = [curr_fit2] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[2.5]]).to(device)
                embedding_array_latentops25 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit25 = get_fitness(init_point, embedding_array_latentops25[0][0],  model.regressor_module)
                fitness_array_latentops25 = [curr_fit25] * cl_args.n_steps
            else:
                y = torch.Tensor([[3.0]]).to(device)
                embedding_array_latentops3  = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit1 = get_fitness(init_point, embedding_array_latentops3[0][0],  model.regressor_module)
                fitness_array_latentops3 = [curr_fit1] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[4.0]]).to(device)
                embedding_array_latentops4 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit15 = get_fitness(init_point, embedding_array_latentops4[0][0],  model.regressor_module)
                fitness_array_latentops4 = [curr_fit15] * cl_args.n_steps


            if cl_args.dataset == 'gifford':
                run_optim_embeddings = [embedding_array_mcmc,
                                        embedding_array_mcmc_cycle,
                                        embedding_array_mcmc_cycle_noP,
                                        embedding_array_hill,
                                        embedding_array_s_hill,
                                        embedding_array_ga,
                                        embedding_array_latentops1[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops15[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops2[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops25[0].cpu().detach().numpy()[0],
                                        ]
                run_optim_fitnesses = [fitness_array_mcmc,
                                            fitness_array_mcmc_cycle,
                                            fitness_array_mcmc_cycle_noP,
                                            fitness_array_hill,
                                            fitness_array_s_hill,
                                            fitness_array_ga,
                                            fitness_array_latentops1,
                                            fitness_array_latentops15,
                                            fitness_array_latentops2,
                                            fitness_array_latentops25
                                    ]
            else:
                run_optim_embeddings = [embedding_array_mcmc,
                                        embedding_array_mcmc_cycle,
                                        embedding_array_mcmc_cycle_noP,
                                        embedding_array_hill,
                                        embedding_array_s_hill,
                                        embedding_array_ga,
                                        embedding_array_latentops3[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops4[0].cpu().detach().numpy()[0],
                                        ]
                run_optim_fitnesses = [fitness_array_mcmc,
                                            fitness_array_mcmc_cycle,
                                            fitness_array_mcmc_cycle_noP,
                                            fitness_array_hill,
                                            fitness_array_s_hill,
                                            fitness_array_ga,
                                            fitness_array_latentops3,
                                            fitness_array_latentops4,
                                        ]
            final_embeddings.append(run_optim_embeddings)

            for alg_indx, (embed, fit) in enumerate(zip(run_optim_embeddings, run_optim_fitnesses )):
                optim_embedding_traj_array[run_indx, alg_indx] = embed
                optim_fitness_traj_array[run_indx, alg_indx] = fit


        # save embeddings
        print("saving embeddings")
        embeddings_save_path = os.path.join(save_dir, "optimized_embeddings.npy")
        np.save(embeddings_save_path, np.array(final_embeddings, dtype=object))

        max_fitness_array = optim_fitness_traj_array[:, :,  -1] # n_init x n_algo array
        utils.plot_boxplot(max_fitness_array, optim_algo_names,
                    wandb_logger=wandb_logger,
                    save_path= save_dir + f'max_fitness_boxplot.png')

        # log max fitness values
        # optim_fitness_traj_array shape: n_inits x n_algos x n_steps
        per_algo_fitness_values = optim_fitness_traj_array.transpose(1,0,2).reshape(len(optim_algo_names), -1)

        for name, fit_vals in zip(optim_algo_names, per_algo_fitness_values):
            max_fit_i = fit_vals.max()
            wandb_logger.experiment.log({f'Max Fitness for {name} Runs': max_fit_i})


        endpoint_embed_array = optim_embedding_traj_array.transpose(1,0,2,3)


        emb_pca_coords = utils.plot_embedding(embeddings, train_targs,
                            wandb_logger=wandb_logger,
                            save_path=save_dir + 'original_fitness_lanscape_pca.png' )
        
        # ------------------------------------------------
        # ------------------------------------------------
        # EVAL Optimization
        # ------------------------------------------------
        # ------------------------------------------------
        
        eval_df = get_evaluation(embeddings_save_path, model, model_ppl, cl_args.dataset)
        csv_save_path = os.path.join(save_dir, "optimization_evaluation.csv")
        eval_df.to_csv(csv_save_path)


        # ------------------------------------------------
        # Unconditional Generation
        # ------------------------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'vae' in model_name:
            number_samples = 100
            with torch.no_grad():
                embeddings = torch.randn(number_samples, cl_args.latent_dim).to(device) 
            embeddings = embeddings.cpu().numpy()
        else:
            model.eval()
            model.to(device)
            with torch.no_grad():
                number_samples = 100
                embeddings = model.ddpm.sample(number_samples,(cl_args.latent_dim,), device)#.cpu()

            embeddings = embeddings.cpu().numpy()

        print("saving embeddings")
        uncond_embeddings_save_path = os.path.join(save_dir, "unconditional_generation_embeddings.npy")
        np.save(uncond_embeddings_save_path, np.array(embeddings, dtype=object))

        uncond_eval_df = get_evaluation(uncond_embeddings_save_path, model, model_ppl, cl_args.dataset)
        csv_save_path = os.path.join(save_dir, "unconditional_generation_evaluation.csv")
        uncond_eval_df.to_csv(csv_save_path)

        # ------------------------------------------------
        # Conditional Generation
        # ------------------------------------------------
        model.eval()
        if 'vae' in model_name:
            number_samples = 100
            with torch.no_grad():
                embeddings = torch.randn(number_samples, cl_args.latent_dim).to(device) 
            embeddings = embeddings.cpu().numpy()
        else:
            model.eval()
            model.to(device)
            with torch.no_grad():
                number_samples = 100
                embeddings = model.ddpm.sample(number_samples,(cl_args.latent_dim,), device)#.cpu()

            embeddings = embeddings.cpu().numpy()
        
        final_embeddings = []

        # model.to('cpu')

        for run_indx, init_indx in enumerate(embeddings):

            init_point = embeddings[run_indx].copy()

            # MCMC
            # print("\n")
            embedding_array_mcmc, fitness_array_mcmc = optim_algs.metropolisMCMC_embedding(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                            delta=cl_args.delta,
                                                                                            N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_mcmc.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc[0][:10]))

            # print("\n")
            embedding_array_mcmc_cycle, fitness_array_mcmc_cycle = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                                oracle=model.regressor_module,
                                                                                                model=model,
                                                                                                delta=cl_args.delta,
                                                                                                N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_mcmc_cycle.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc_cycle[0][:10]))


            # print("\n")
            embedding_array_mcmc_cycle_noP, fitness_array_mcmc_cycle_noP = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                                oracle=model.regressor_module,
                                                                                                model=model,
                                                                                                N_steps=n_steps,
                                                                                                delta=cl_args.delta,
                                                                                                perturbation=False)
            # print(f'shape of output embedding array: {embedding_array_mcmc_cycle_noP.shape}')
            # print('init embed from output: {}'.format(embedding_array_mcmc_cycle_noP[0][:10]))


            # # Hill climbing
            # print("\n")
            embedding_array_hill, fitness_array_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                            dataset_embeddings=embeddings,
                                                                                            N_steps=n_steps)
            # print(f'shape of output embedding array: {embedding_array_hill.shape}')
            # print('init embed from output: {}'.format(embedding_array_hill[0][:10]))

            # # Stochastic climbing
            # print("\n")
            embedding_array_s_hill, fitness_array_s_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding = init_point.copy(),
                                                                                                oracle = model.regressor_module,
                                                                                                dataset_embeddings=embeddings,
                                                                                                N_steps=n_steps,
                                                                                                stochastic=True)
            # print(f'shape of output embedding array: {embedding_array_s_hill.shape}')
            # print('init embed from output: {}'.format(embedding_array_s_hill[0][:10]))

            # # Gradient Ascent
            # print("\n")
            embedding_array_ga, fitness_array_ga = optim_algs.grad_ascent(initial_embedding = init_point.copy(),
                                                                        model=model,
                                                                        N_steps=n_steps,
                                                                        lr=0.1)

            # print(f'shape of output embedding array: {embedding_array_ga.shape}')
            # print('init embed from output: {}'.format(embedding_array_ga[0][:10]))


            ########################
            # LatentOps Optimization
            ########################

            # print initial fitness
            initial_embedding = init_point.copy()   
            curr_fit = get_fitness(init_point, initial_embedding,  model.regressor_module)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            classifier_list = [model.regressor_module]
            classifier = CCF(classifier_list, cl_args)
            classifier.to(device)
            y = torch.Tensor([[cl_args.y_label]]).to(device)
            init_point_tensor = torch.from_numpy(init_point).to(device).unsqueeze(dim=0)
            init_point_tensor = init_point_tensor.repeat(2, 1)


            z = init_point_tensor

            if cl_args.dataset == 'gifford':
                y = torch.Tensor([[1.0]]).to(device)
                embedding_array_latentops1  = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit1 = get_fitness(init_point, embedding_array_latentops1[0][0],  model.regressor_module)
                fitness_array_latentops1 = [curr_fit1] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[1.5]]).to(device)
                embedding_array_latentops15 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit15 = get_fitness(init_point, embedding_array_latentops15[0][0],  model.regressor_module)
                fitness_array_latentops15 = [curr_fit15] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[2.0]]).to(device)
                embedding_array_latentops2 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit2 = get_fitness(init_point, embedding_array_latentops2[0][0],  model.regressor_module)
                fitness_array_latentops2 = [curr_fit2] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[2.5]]).to(device)
                embedding_array_latentops25 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit25 = get_fitness(init_point, embedding_array_latentops25[0][0],  model.regressor_module)
                fitness_array_latentops25 = [curr_fit25] * cl_args.n_steps
            else:
                y = torch.Tensor([[3.0]]).to(device)
                embedding_array_latentops3  = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit1 = get_fitness(init_point, embedding_array_latentops3[0][0],  model.regressor_module)
                fitness_array_latentops3 = [curr_fit1] * cl_args.n_steps

                classifier.to(device)
                y = torch.Tensor([[4.0]]).to(device)
                embedding_array_latentops4 = sample_q_ode(classifier, y, device = device, model_name = cl_args.model_paths, **ode_kwargs, **ld_kwargs, z_k = z)
                curr_fit15 = get_fitness(init_point, embedding_array_latentops4[0][0],  model.regressor_module)
                fitness_array_latentops4 = [curr_fit15] * cl_args.n_steps

            if cl_args.dataset == 'gifford':
                run_optim_embeddings = [embedding_array_mcmc,
                                        embedding_array_mcmc_cycle,
                                        embedding_array_mcmc_cycle_noP,
                                        embedding_array_hill,
                                        embedding_array_s_hill,
                                        embedding_array_ga,
                                        embedding_array_latentops1[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops15[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops2[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops25[0].cpu().detach().numpy()[0],
                                        ]
                run_optim_fitnesses = [fitness_array_mcmc,
                                            fitness_array_mcmc_cycle,
                                            fitness_array_mcmc_cycle_noP,
                                            fitness_array_hill,
                                            fitness_array_s_hill,
                                            fitness_array_ga,
                                            fitness_array_latentops1,
                                            fitness_array_latentops15,
                                            fitness_array_latentops2,
                                            fitness_array_latentops25
                                    ]
            else:
                run_optim_embeddings = [embedding_array_mcmc,
                                        embedding_array_mcmc_cycle,
                                        embedding_array_mcmc_cycle_noP,
                                        embedding_array_hill,
                                        embedding_array_s_hill,
                                        embedding_array_ga,
                                        embedding_array_latentops3[0].cpu().detach().numpy()[0],
                                        embedding_array_latentops4[0].cpu().detach().numpy()[0],
                                        ]
                run_optim_fitnesses = [fitness_array_mcmc,
                                            fitness_array_mcmc_cycle,
                                            fitness_array_mcmc_cycle_noP,
                                            fitness_array_hill,
                                            fitness_array_s_hill,
                                            fitness_array_ga,
                                            fitness_array_latentops3,
                                            fitness_array_latentops4,
                                        ]
            final_embeddings.append(run_optim_embeddings)

        # save embeddings
        cond_embeddings_save_path = os.path.join(save_dir, "conditional_generation_embeddings.npy")
        np.save(cond_embeddings_save_path, np.array(final_embeddings, dtype=object))

        cond_eval_df = get_evaluation(cond_embeddings_save_path, model, model_ppl, cl_args.dataset)
        csv_save_path = os.path.join(save_dir, "conditional_generation_evaluation.csv")
        cond_eval_df.to_csv(csv_save_path)


        # ------------------------------------------------
        # Visualization (Latent Space)
        # ------------------------------------------------
        train_reps, _, train_targs = data.train_split.tensors 
        train_outputs, train_hrep, train_yhat = eval_utils.get_diff_model_outputs(model, train_reps, device)
        emb_coords = PCA(n_components=2).fit_transform(train_hrep)
        fitness = train_targs

        if cl_args.dataset == 'gifford':
            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(emb_coords[:,0], emb_coords[:,1], c='grey', s=3)
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=-2, fitness<-1))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='red', s=3, label='-2 <= fitness < -1')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=-1, fitness<-0.5))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='orange', s=3, label='-1 <= fitness < -0.5')
            interval_2 = emb_coords[np.where(np.logical_and(fitness>=-0.5, fitness<0))[0]]
            ax.scatter(interval_2[:,0], interval_2[:,1], c='yellow', s=3, label='0.5 <= fitness < 0')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=0, fitness<0.5))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='green', s=3, label='0 <= fitness < 0.5')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=0.5, fitness<1))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='blue', s=3, label='0.5 <= fitness < 1')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=1, fitness<2))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='purple', s=3, label='1 <= fitness < 2')
            ax.legend(fontsize=12)
        else:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(emb_coords[:,0], emb_coords[:,1], c='grey', s=3)
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=1, fitness<1.5))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='red', s=3, label='1 <= fitness < 1.5')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=1.5, fitness<2))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='orange', s=3, label='1.5 <= fitness < 2')
            interval_2 = emb_coords[np.where(np.logical_and(fitness>=2, fitness<2.5))[0]]
            ax.scatter(interval_2[:,0], interval_2[:,1], c='yellow', s=3, label='2 <= fitness < 2.5')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=2.5, fitness<3))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='green', s=3, label='2.5 <= fitness < 3')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=3, fitness<3.5))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='blue', s=3, label='3 <= fitness < 3.5')
            interval_1 = emb_coords[np.where(np.logical_and(fitness>=3.5, fitness<4))[0]]
            ax.scatter(interval_1[:,0], interval_1[:,1], c='purple', s=3, label='3.5 <= fitness < 4')
            ax.legend(fontsize=12)
        plt.savefig(os.path.join(save_dir, 'latent_colored.jpg'))
        wandb_logger.experiment.log({f'Colored Latent Space': wandb.Image(plt)})

        # ------------------------------------------------
        # Visualization (Optimization Point)
        # ------------------------------------------------

        if cl_args.dataset == 'gifford':
            new_embeddings = np.load(embeddings_save_path, allow_pickle=True)
            new_emb1 = np.array([lst.tolist() for lst in new_embeddings[:, 6]])#.shape
            new_emb15 = np.array([lst.tolist() for lst in new_embeddings[:, 7]])
            new_emb2 = np.array([lst.tolist() for lst in new_embeddings[:, 8]])
            new_emb25 = np.array([lst.tolist() for lst in new_embeddings[:, 9]])

            all_embed2 = np.concatenate([train_hrep, new_emb1, new_emb15, new_emb2, new_emb25], 0)
            all_coords2 = PCA(n_components=2).fit_transform(all_embed2)

            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(all_coords2[:train_hrep.shape[0],0], all_coords2[:train_hrep.shape[0],1], c='grey', s=3)
            ax.scatter(all_coords2[train_hrep.shape[0]:train_hrep.shape[0]+num_inits*1,0], all_coords2[train_hrep.shape[0]:train_hrep.shape[0]+num_inits*1,1], c='red', s=10, label='y=1')
            ax.scatter(all_coords2[train_hrep.shape[0]+num_inits*1:train_hrep.shape[0]+num_inits*2,0], all_coords2[train_hrep.shape[0]+num_inits*1:train_hrep.shape[0]+num_inits*2,1], c='yellow', s=10, label='y=1.5')
            ax.scatter(all_coords2[train_hrep.shape[0]+num_inits*2:train_hrep.shape[0]+num_inits*3,0], all_coords2[train_hrep.shape[0]+num_inits*2:train_hrep.shape[0]+num_inits*3,1], c='green', s=10, label='y=2')
            ax.scatter(all_coords2[train_hrep.shape[0]+num_inits*3:train_hrep.shape[0]+num_inits*4,0], all_coords2[train_hrep.shape[0]+num_inits*3:train_hrep.shape[0]+num_inits*4,1], c='blue', s=10, label='y=2.5')
            plt.legend(fontsize=14)
            plt.savefig(os.path.join(save_dir, 'test_optimization.jpg'))
            wandb_logger.experiment.log({f'Colored Optimized Seqs': wandb.Image(plt)})
        else:
            new_embeddings = np.load(embeddings_save_path, allow_pickle=True)
            new_emb3 = np.array([lst.tolist() for lst in new_embeddings[:, 6]])#.shape
            new_emb4 = np.array([lst.tolist() for lst in new_embeddings[:, 7]])

            all_embed2 = np.concatenate([train_hrep, new_emb3, new_emb4], 0)
            all_coords2 = PCA(n_components=2).fit_transform(all_embed2)

            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(all_coords2[:train_hrep.shape[0],0], all_coords2[:train_hrep.shape[0],1], c='grey', s=3)
            ax.scatter(all_coords2[train_hrep.shape[0]:train_hrep.shape[0]+num_inits*1,0], all_coords2[train_hrep.shape[0]:train_hrep.shape[0]+num_inits*1,1], c='red', s=10, label='y=3')
            ax.scatter(all_coords2[train_hrep.shape[0]+num_inits*1:train_hrep.shape[0]+num_inits*2,0], all_coords2[train_hrep.shape[0]+num_inits*1:train_hrep.shape[0]+num_inits*2,1], c='yellow', s=10, label='y=4')
            plt.legend(fontsize=14)
            plt.savefig(os.path.join(save_dir, 'test_optimization.jpg'))
            wandb_logger.experiment.log({f'Colored Optimized Seqs': wandb.Image(plt)})

    all_eval_results_df = pd.DataFrame(all_eval_results)
    df_save_path = os.path.join(cl_args.model_path, "all_eval_results.csv")
    all_eval_results_df.to_csv(df_save_path)
