

import torch
from torch import nn
from torch.nn import functional as F


import argparse
import wandb
import math

from codes.nn.bneck import BaseBottleneck

from codes.nn.base import BaseModel
from codes.nn.convolutional import Block
from codes.nn.transformers import PositionalEncoding, TransformerEncoder
from codes.nn.auxnetwork import str2auxnetwork
from typing import Dict, Tuple

import torch.nn.init as init

# --------------------
# ReLSO model
# --------------------

def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)

class relso1(BaseModel):
    def __init__(self, hparams):
        super(relso1, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()
        # self.hparams = hparams

        # model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = 4
        self.src_mask = None
        self.bz = hparams.batch_size

        self.lr = hparams.lr
        self.model_name = "relso1"

        self.alpha = hparams.alpha_val
        self.gamma = hparams.gamma_val

        self.sigma = hparams.sigma_val

        try:
            self.eta = hparams.eta_val
        except:
            self.eta = 1.0

        try:
            self.interp_samping = hparams.interp_samp
        except:
            self.interp_samping = True

        try:
            self.negative_sampling = hparams.neg_samp
        except:
            self.negative_sampling = True

        try:
            self.neg_size = hparams.neg_size
            self.neg_floor = hparams.neg_floor
            self.neg_weight = hparams.neg_weight
            self.neg_focus = hparams.neg_focus
            self.neg_norm = hparams.neg_norm
        except:
            self.neg_focus = False

        self.dyn_neg_bool = False  # set False as default

        try:
            self.interp_size = hparams.interp_size
            self.interp_weight = hparams.interp_weight

        except:
            pass
        self.interp_inds = None
        self.dyn_interp_bool = False

        try:
            self.wl2norm = hparams.wl2norm
        except:
            self.wl2norm = False

        self.g_opt_step = 0

        self.seq_len = hparams.seq_len

        # The embedding input dimension may need to be changed depending on the size of our dictionary
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)

        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim, max_len=self.seq_len
        )

        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1), nn.Softmax(dim=1)
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )
        # make decoder)
        self._build_decoder(hparams)

        # for los and gradient checking
        self.z_rep = None

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.embedding_dim, self.latent_dim)

        # self.bottleneck = BaseBottleneck(self.embedding_dim, self.latent_dim)
        aux_params = {"latent_dim": self.latent_dim, "probs": hparams.probs}
        aux_hparams = argparse.Namespace(**aux_params)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            self.regressor_module = auxnetwork(aux_hparams)
        except:
            print('aux network loading failed - defaulting to SpectralRegressor')
            auxnetwork = str2auxnetwork("spectral")
            self.regressor_module = auxnetwork(aux_hparams)

    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _build_decoder(self, hparams):
        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            Block(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

        print(dec_layers)

    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            )
        # self.embed gives output (batch_size,sequence_length,num_features)
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)

        return output_embed

    def encode(self, batch):
        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed)  # output should be B x S x 1
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.bottleneck_module(z_rep)

        return z_rep

    def decode(self, z_rep):

        h_rep = z_rep  # B x 1 X L

        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)

            h_rep = layer(h_rep)

        return h_rep

    def predict_fitness(self, batch):

        z = self.encode(batch)

        y_hat = self.regressor_module(z)

        return y_hat

    def predict_fitness_from_z(self, z_rep):

        # to regain the batch dimension
        if len(z_rep) == 1:
            z_rep = z_rep.unsqueeze(0)

        y_hat = self.regressor_module(z_rep)

        return y_hat

    def interpolation_sampling(self, z_rep):
        """
        get interpolations between z_reps in batch
        interpolations between z_i and its nearest neighbor
        Args:
            z_rep ([type]): [description]
        Returns:
            [type]: [description]
        """
        z_dist_mat = self.pairwise_l2(z_rep, z_rep)
        k_val = min(len(z_rep), 2)
        _, z_nn_inds = z_dist_mat.topk(k_val, largest=False)
        z_nn_inds = z_nn_inds[:, 1]

        z_nn = z_rep[z_nn_inds]
        z_interp = (z_rep + z_nn) / 2

        # use 10 % of batch size
        subset_inds = torch.randperm(len(z_rep), device=self.device)[: self.interp_size]

        sub_z_interp = z_interp[subset_inds]
        sub_nn_inds = z_nn_inds[subset_inds]

        self.interp_inds = torch.cat(
            (subset_inds.unsqueeze(1), sub_nn_inds.unsqueeze(1)), dim=1
        )

        return sub_z_interp

    def add_negative_samples(
        self,
    ):

        max2norm = torch.norm(self.z_rep, p=2, dim=1).max()
        wandb.log({"max2norm": max2norm})
        rand_inds = torch.randperm(len(self.z_rep))
        if self.neg_focus:
            neg_z = (
                0.5 * torch.randn_like(self.z_rep)[: self.neg_size]
                + self.z_rep[rand_inds][: self.neg_size]
            )
            neg_z = neg_z / torch.norm(neg_z, 2, dim=1).reshape(-1, 1)
            neg_z = neg_z * (max2norm * self.neg_norm)

        else:
            center = self.z_rep.mean(0, keepdims=True)
            dist_set = self.z_rep - center

            # gets maximally distant rep from center
            dist_sort = torch.norm(dist_set, 2, dim=1).reshape(-1, 1).sort().indices[-1]
            max_dist = dist_set[dist_sort]
            adj_dist = self.neg_norm * max_dist.repeat(len(self.z_rep), 1) - dist_set
            neg_z = self.z_rep + adj_dist
            neg_z = neg_z[rand_inds][: self.neg_size]

        # else:
        # neg_z = torch.randn_like(self.z_rep)[:self.neg_size]

        return neg_z

    def forward(self, batch):

        z_rep = self.encode(batch)
        self.z_rep = z_rep

        # interpolative samping
        # ---------------------------------------
        # only do interpolative sampling if batch size is expected size
        self.dyn_interp_bool = self.interp_samping and z_rep.size(0) == self.bz
        if self.dyn_interp_bool:
            z_i_rep = self.interpolation_sampling(z_rep)
            interp_z_rep = torch.cat((z_rep, z_i_rep), 0)

            x_hat = self.decode(interp_z_rep)

        else:

            x_hat = self.decode(z_rep)

        # negative sampling
        # ---------------------------------------
        self.dyn_neg_bool = self.negative_sampling and z_rep.size(0) == self.bz
        if self.dyn_neg_bool:
            z_n_rep = self.add_negative_samples()
            neg_z_rep = torch.cat((z_rep, z_n_rep), 0)

            y_hat = self.regressor_module(neg_z_rep)
        else:
            y_hat = self.regressor_module(z_rep)

        # safety precaution: not sure if I can
        # overwrite variables used in autograd tape

        return [x_hat, y_hat], z_rep

    def pairwise_l2(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, 2).sum(2)

        return dist

    def loss_function(self, predictions, targets, valid_step=False):

        # unpack everything
        x_hat, y_hat = predictions
        x_true, y_true = targets

        if self.dyn_interp_bool:
            recon_x = x_hat[: -self.interp_size]
        else:
            recon_x = x_hat

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(22, device=self.device)
        ce_loss_weights[21] = 0.8

        ae_loss = nn.CrossEntropyLoss(weight=ce_loss_weights)(recon_x, x_true)
        ae_loss = self.gamma * ae_loss

        if self.dyn_neg_bool:
            pred_y = y_hat[: -self.neg_size]
            extend_y = y_hat[-self.neg_size :]
        else:
            pred_y = y_hat

        # enrichment pred loss
        reg_loss = nn.MSELoss()(pred_y.flatten(), y_true.flatten())
        reg_loss = self.alpha * reg_loss

        # interpolation loss
        # z_dist_mat = self.pairwise_l2(self.z_rep, self.z_rep)
        if self.dyn_interp_bool:
            seq_preds = (
                F.gumbel_softmax(x_hat, tau=1, dim=1, hard=True)
                .transpose(1, 2)
                .flatten(1, 2)
            )
            seq_dist_mat = torch.cdist(seq_preds, seq_preds, p=1)

            # rint(f' bs {self.bz}\t z_rep: {self.z_rep.size(0)}')

            ext_inds = torch.arange(self.bz, self.bz + self.interp_size)
            tr_dists = seq_dist_mat[self.interp_inds[:, 0], self.interp_inds[:, 1]]
            inter_dist1 = seq_dist_mat[ext_inds, self.interp_inds[:, 0]]
            inter_dist2 = seq_dist_mat[ext_inds, self.interp_inds[:, 1]]

            interp_loss = (0.5 * (inter_dist1 + inter_dist2) - 0.5 * tr_dists).mean()
            interp_loss = max(0, interp_loss) * self.interp_weight
        else:
            interp_loss = 0.0

        # negative sampling loss
        if self.dyn_neg_bool:
            neg_targets = (
                torch.ones((self.neg_size), device=self.device) * self.neg_floor
            )

            # print(f'neg_targets: {neg_targets}')

            neg_loss = nn.MSELoss()(extend_y.flatten(), neg_targets.flatten())
            neg_loss = neg_loss * self.neg_weight

            # print(f'neg_loss: {neg_loss}')
        else:
            neg_loss = 0.0

        # RAE L_z loss
        # only penalize real zs
        zrep_l2_loss = 0.5 * torch.norm(self.z_rep, 2, dim=1) ** 2

        if self.wl2norm:
            y_true_shift = y_true + torch.abs(y_true.min())
            w_fit_zrep = nn.ReLU()(y_true_shift / y_true_shift.sum())
            zrep_l2_loss = torch.dot(zrep_l2_loss.flatten(), w_fit_zrep.flatten())
        else:
            zrep_l2_loss = zrep_l2_loss.mean()

        zrep_l2_loss = zrep_l2_loss * self.eta

        total_loss = ae_loss + reg_loss + zrep_l2_loss + interp_loss + neg_loss

        mloss_dict = {
            "ae_loss": ae_loss,
            "zrep_l2_loss": zrep_l2_loss,
            "interp_loss": interp_loss,
            "neg samp loss": neg_loss,
            "reg_loss": reg_loss,
            "loss": total_loss,
        }

        return total_loss, mloss_dict

# Relso Diffusion
class RelsoDiffusion(BaseModel):
    def __init__(self, hparams):
        super(RelsoDiffusion, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()
        # self.hparams = hparams

        # model ddpm
        self.ddpm = DDPM(eps_model=MLPSkipNet(hparams.latent_dim), betas=(1e-4, 0.02), n_T=hparams.nt,)
        self.ddpm_weight = hparams.ddpm_weight
        self.joint_regressor = hparams.joint_regressor

        # model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = 4
        self.src_mask = None
        self.bz = hparams.batch_size

        self.lr = hparams.lr
        self.model_name = "relso1"

        self.alpha = hparams.alpha_val
        self.gamma = hparams.gamma_val

        self.sigma = hparams.sigma_val

        try:
            self.eta = hparams.eta_val
        except:
            self.eta = 1.0

        try:
            self.interp_samping = hparams.interp_samp
        except:
            self.interp_samping = True

        try:
            self.negative_sampling = hparams.neg_samp
        except:
            self.negative_sampling = True

        try:
            self.neg_size = hparams.neg_size
            self.neg_floor = hparams.neg_floor
            self.neg_weight = hparams.neg_weight
            self.neg_focus = hparams.neg_focus
            self.neg_norm = hparams.neg_norm
        except:
            self.neg_focus = False

        self.dyn_neg_bool = False  # set False as default

        try:
            self.interp_size = hparams.interp_size
            self.interp_weight = hparams.interp_weight

        except:
            pass
        self.interp_inds = None
        self.dyn_interp_bool = False

        try:
            self.wl2norm = hparams.wl2norm
        except:
            self.wl2norm = False

        self.g_opt_step = 0

        self.seq_len = hparams.seq_len

        # The embedding input dimension may need to be changed depending on the size of our dictionary
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)

        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim, max_len=self.seq_len
        )

        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1), nn.Softmax(dim=1)
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )
        # make decoder)
        self._build_decoder(hparams)

        # for los and gradient checking
        self.z_rep = None

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.embedding_dim, self.latent_dim)

        # self.bottleneck = BaseBottleneck(self.embedding_dim, self.latent_dim)
        aux_params = {"latent_dim": self.latent_dim, "probs": hparams.probs}
        aux_hparams = argparse.Namespace(**aux_params)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            self.regressor_module = auxnetwork(aux_hparams)
        except:
            print('aux network loading failed - defaulting to SpectralRegressor')
            auxnetwork = str2auxnetwork("spectral")
            self.regressor_module = auxnetwork(aux_hparams)

    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _build_decoder(self, hparams):
        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            Block(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

        print(dec_layers)

    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            )
        # self.embed gives output (batch_size,sequence_length,num_features)
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)

        return output_embed

    def encode(self, batch):
        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed)  # output should be B x S x 1
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        mu, logvar = self.bottleneck_module(z_rep)
        logvar = torch.log(torch.ones_like(logvar) * 0.008)

        return mu, logvar

    def decode(self, z_rep):

        h_rep = z_rep  # B x 1 X L

        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)

            h_rep = layer(h_rep)

        return h_rep

    def predict_fitness(self, batch):

        z = self.encode(batch)

        y_hat = self.regressor_module(z)

        return y_hat

    def predict_fitness_from_z(self, z_rep):

        # to regain the batch dimension
        if len(z_rep) == 1:
            z_rep = z_rep.unsqueeze(0)

        y_hat = self.regressor_module(z_rep)

        return y_hat

    def interpolation_sampling(self, z_rep):
        """
        get interpolations between z_reps in batch
        interpolations between z_i and its nearest neighbor
        Args:
            z_rep ([type]): [description]
        Returns:
            [type]: [description]
        """
        z_dist_mat = self.pairwise_l2(z_rep, z_rep)
        k_val = min(len(z_rep), 2)
        _, z_nn_inds = z_dist_mat.topk(k_val, largest=False)
        z_nn_inds = z_nn_inds[:, 1]

        z_nn = z_rep[z_nn_inds]
        z_interp = (z_rep + z_nn) / 2

        # use 10 % of batch size
        subset_inds = torch.randperm(len(z_rep), device=self.device)[: self.interp_size]

        sub_z_interp = z_interp[subset_inds]
        sub_nn_inds = z_nn_inds[subset_inds]

        self.interp_inds = torch.cat(
            (subset_inds.unsqueeze(1), sub_nn_inds.unsqueeze(1)), dim=1
        )

        return sub_z_interp

    def add_negative_samples(
        self,
    ):

        max2norm = torch.norm(self.z_rep, p=2, dim=1).max()
        wandb.log({"max2norm": max2norm})
        rand_inds = torch.randperm(len(self.z_rep))
        if self.neg_focus:
            neg_z = (
                0.5 * torch.randn_like(self.z_rep)[: self.neg_size]
                + self.z_rep[rand_inds][: self.neg_size]
            )
            neg_z = neg_z / torch.norm(neg_z, 2, dim=1).reshape(-1, 1)
            neg_z = neg_z * (max2norm * self.neg_norm)

        else:
            center = self.z_rep.mean(0, keepdims=True)
            dist_set = self.z_rep - center

            # gets maximally distant rep from center
            dist_sort = torch.norm(dist_set, 2, dim=1).reshape(-1, 1).sort().indices[-1]
            max_dist = dist_set[dist_sort]
            adj_dist = self.neg_norm * max_dist.repeat(len(self.z_rep), 1) - dist_set
            neg_z = self.z_rep + adj_dist
            neg_z = neg_z[rand_inds][: self.neg_size]

        return neg_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, batch):

        mu, logvar = self.encode(batch)
        z_rep = self.reparameterize(mu, logvar)
        self.z_rep = z_rep

        x_hat = self.decode(z_rep)

        y_hat = self.regressor_module(z_rep)

        # return [x_hat, y_hat], z_rep
        return [x_hat, z_rep, mu, logvar, y_hat], z_rep

    def pairwise_l2(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, 2).sum(2)

        return dist

    def loss_function(self, predictions, targets, valid_step=False):

        # unpack everything
        x_hat, z_rep, mu, logvar, y_hat = predictions
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        x_true, y_true = targets

        recon_x = x_hat

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(22, device=self.device)
        ce_loss_weights[21] = 0.8

        vae_loss = nn.CrossEntropyLoss(reduction='none', weight=ce_loss_weights)(recon_x, x_true)
        vae_loss = vae_loss.mean(axis=1)

        ddpm_loss, loss_weight = self.ddpm.forward_new(z_rep, mu)

        reg_loss = nn.MSELoss()(y_hat.flatten(), y_true.flatten())


        if self.ddpm_weight > 0:
            total_loss = (1/(loss_weight * self.ddpm.n_T)  * vae_loss).mean() + self.ddpm_weight *ddpm_loss.mean() 
        else:
            total_loss = vae_loss.mean() + 0.0* ddpm_loss.mean()

        if self.joint_regressor:
            total_loss = total_loss + reg_loss
        
        mloss_dict = {
            "ae_loss": vae_loss.mean(),
            "ddpm_loss":ddpm_loss.mean(), 
            "reg_loss": reg_loss,
            "loss": total_loss,
        }

        return total_loss, mloss_dict


def ddpm_schedules(beta1: float, beta2: float, T: int)-> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1

    beta_t = beta_t*0 + 0.008
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    sqrta = torch.sqrt(alpha_t)
    oneover_sqrta = 1 / sqrta
    mab = 1 - alphabar_t
    sqrtmab = torch.sqrt(mab)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    loss_weight = 0.5 * beta_t / (alpha_t * mab)
    sigma = sqrt_beta_t
    sqrtmab_over_sqrtab = sqrtmab / sqrtab
    sigma_diff = sqrtmab_over_sqrtab[1:] - sqrtmab_over_sqrtab[:-1]
    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,  
        "sqrta": sqrta,
        "oneover_sqrta": oneover_sqrta,  
        "sqrt_beta_t": sqrt_beta_t,  
        "alphabar_t": alphabar_t,  
        "sqrtab": sqrtab,  
        "mab": mab,
        "sqrtmab": sqrtmab,  
        "mab_over_sqrtmab": mab_over_sqrtmab_inv, 
        "sigma" : sigma,
        'loss_weight':loss_weight,
        'diff_sigma': sigma_diff
    }


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(reduction='none'),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward_new(self, x: torch.Tensor, mu) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(device=x.device) # t ~ Uniform(0, n_T-1)  before: (1, n_T-1)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)   
        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

        eps_0 = self.eps_model(x_t, _ts / self.n_T)
        loss =  self.criterion(eps, eps_0).mean(1)  # compute loss 64 * 1
        mask_1 = (_ts == 1)
        if mask_1.any():
            loss_z0 = self.alpha_t[1]/self.beta_t[1] * self.criterion(mu, self.oneover_sqrta[1]*(x_t - self.sqrt_beta_t[1]*eps_0)).mean(1)
            loss = torch.where(mask_1,loss_z0, loss) 
        
        return loss,self.loss_weight[_ts, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def add_noise(self, x_i: torch.Tensor, T=None) -> torch.Tensor:
        """
        DDIM forward
        """
        if T == None:
            T = self.n_T
        with torch.no_grad():
            n_sample = x_i.size(0)
            for i in range(0, T):  # 0,T
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)
                x_i = self.sqrta[i+1] * (x_i + (self.sqrtmab[i+1]/self.sqrta[i+1] - self.sqrtmab[i])*eps )
        return x_i

    def add_noise_ddpm(self, x_i: torch.Tensor, step=None):
        noise = torch.randn_like(x_i) 
        x_i = self.sqrtab[step]* x_i + self.sqrtmab[step] * noise
        return x_i

    def add_vpnoise(self, x_i):
        n_sample = x_i.size(0)
        for i in range(1, self.n_T):
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            score = - eps / self.sqrtmab[i]
            x_i = x_i - 0.5*i/self.n_T * self.beta_t[i] * (x_i + score)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i
    
    def sample_new(self, n_sample: int, size, device, fp16=False) -> torch.Tensor:
        dtype_ = torch.half if fp16 else torch.float
        x_i = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)  # x_T ~ N(0, 1)
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device=device,dtype=dtype_)  if i > 1 else 0
            ts_ = torch.tensor(i).to(device=device,dtype=dtype_) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i

    def sample_cond(self, n_sample: int, size, device, classifier, y, scale=500, softmax_logits=False) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        
        scale = -((y.view(-1,1) - 1) * (scale[0]-scale[1]) - scale[1] )
        classifier_scale = scale
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)

                with torch.set_grad_enabled(True):
                    x_in = x_i.detach().requires_grad_(True)
                    logits = classifier.train_step(x_in, ts_)
                    if softmax_logits:
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        tmp = torch.autograd.grad(selected.sum(),x_in)[0]
                    else:  ### log_probs = energy
                        neg_energy = torch.gather(logits, 1, y.view(-1)[:,None]).squeeze() - logits.logsumexp(1)
                        tmp = torch.autograd.grad(neg_energy.sum(), x_in)[0]
                    grad_z = tmp * scale
                ##### DDIM 
                eps = eps - self.sqrtmab[i] * grad_z
                ############  DDPM
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.beta_t[i]*grad_z
                    + self.sqrt_beta_t[i] * z
                )
                #############
        return x_i

    def sample_cond_post(self,x_i, device, classifier, y, scale=[500,200], step=2000, softmax_logits=False) -> torch.Tensor:
        # x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        n_sample = x_i.shape[0]
        size = (x_i.shape[1],)
        scale = -((y.view(-1,1) - 1) * (scale[0]-scale[1]) - scale[1] )
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                ts_ = ts_.repeat(n_sample)
                eps = self.eps_model(x_i, ts_)
                if i <= step:
                    with torch.set_grad_enabled(True):
                        x_in = x_i.detach().requires_grad_(True)
                        logits = classifier.train_step(x_in, ts_ * self.n_T)
                        if softmax_logits: ### log_probs = logsoftmax
                            log_probs = F.log_softmax(logits, dim=-1)
                            selected = log_probs[range(len(logits)), y.view(-1)]
                            tmp = torch.autograd.grad(selected.sum(),x_in)

                        else:  ### log_probs = energy
                            neg_energy = torch.gather(logits, 1, y.view(-1)[:,None]).squeeze() - logits.logsumexp(1)
                            tmp = torch.autograd.grad(neg_energy.sum(), x_in)
                        grad_z = tmp[0] * scale
                    eps = eps - self.sqrtmab[i] * grad_z
                eta = 0.0
                sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                x_i = self.sqrtab[i-1] * ((x_i - self.sqrtmab[i] * eps )/ self.sqrtab[i]) + self.sqrtmab[i-1]* eps 
        return x_i
    def sample_one(self, n_sample: int, size, device, score_flag = 2, T=None, step=None, fp16=False) -> torch.Tensor:
        dtype_ = torch.half if fp16 else torch.float
        if T == None:
            T = self.n_T
        with torch.no_grad():
            x_i = torch.randn( n_sample, *size).to(device=device,dtype=dtype_)  # x_T ~ N(0, 1)
            # This samples accordingly to Algorithm 2. It is exactly the same logic.
            if score_flag == 4:  ## shperical interpolation
                z1 = torch.randn_like(x_i)
                theta = torch.arccos((x_i * z1).sum(1)/(torch.norm(z1,dim=1)*torch.norm(x_i,dim=1)))
                sin_theta = torch.sin(theta)
                jj = 0.5
                tmp1 = torch.matmul(torch.diag(torch.sin((1-jj)*theta)/sin_theta) , z1)
                tmp2 = torch.matmul(torch.diag(torch.sin(jj*theta)/sin_theta) ,x_i)
                x_i = tmp1  +  tmp2
            if score_flag == 6:
                z1 = torch.randn_like(x_i).to(device=device,dtype=dtype_)
                x_i = (z1  +  x_i) * 0.5
            elif score_flag == 7:
                step_list = list(range(T,0,-T//step))
                cnt_step = 0
            for i in range(T, 0, -1):
                z = torch.randn(n_sample, *size).to(device=device,dtype=dtype_) if i > 1 else 0
                if score_flag != 7:
                    ts_ = torch.tensor(i).to(device=device,dtype=dtype_) / self.n_T
                    ts_ = ts_.repeat(n_sample)
                    eps = self.eps_model(x_i,ts_)
                
                if score_flag == 0: # score model
                    score = - eps / self.sqrtmab[i]
                    x_i = (
                        self.oneover_sqrta[i] * (x_i + self.beta_t[i] * score) + self.sqrt_beta_t[i] * z
                    )
                elif score_flag == 1: # DDPM
                    x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                    )
                elif score_flag == 2: # DDIM
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)

                elif score_flag == 3: # VP-SDE -> ODE
                    x_i = x_i - 1./self.n_T * 0.5*self.beta_t[i]*(eps * self.sqrtmab[i] - x_i) 
                elif score_flag == 4: # DDIM with Interpolation
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)

                elif score_flag == 6: # DDIM with linear Interpolation
                    eta = 0.0
                    sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                    x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
                elif score_flag == 5: # VP  ODE
                    score = - eps / self.sqrtmab[i]
                    x_i = (2 - torch.sqrt(1-self.beta_t[i])) * x_i + 0.5 * self.beta_t[i] * score
                elif score_flag == 7 and i in step_list[:-1]: # DDIM with less steps
                    ts_ = torch.tensor(i).to(x_i.device) / self.n_T
                    ts_ = ts_.repeat(n_sample)
                    eps = self.eps_model(x_i,ts_)
                    eta = 0.0
                    next_step = step_list[cnt_step+1]
                    x_i = self.sqrtab[next_step] * (x_i - self.sqrtmab[i]*eps)/self.sqrtab[i] + self.sqrtmab[next_step] * eps
                    cnt_step+= 1
        return x_i
    def sample_posterior(self, x_i, device, score_flag=2, T=None) -> torch.Tensor:
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        n_sample = x_i.size(0)
        if T == None:
            T = self.n_T
        for i in range(T, 0, -1):
            z = torch.randn_like(x_i).to(device) if i > 2 else 0
            ts_ = torch.tensor(i).to(x_i.device) / self.n_T
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i, ts_)
            if score_flag == 0: # score model
                score = - eps / self.sqrtmab[i]
                x_i = (
                    self.oneover_sqrta[i] * (x_i + self.beta_t[i] * score) + self.sqrt_beta_t[i] * z
                )
            elif score_flag == 1: # DDPM
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
            elif score_flag == 2: # DDIM
                eta = 0.0
                sigma_ = eta * self.sqrtmab[i-1] / self.sqrtmab[i] * self.sqrt_beta_t[i]
                x_i = ( self.oneover_sqrta[i] * (x_i - (self.sqrtmab[i] - self.sqrta[i] * torch.sqrt(self.mab[i-1] - sigma_**2)) * eps ) + sigma_ * z)
            elif score_flag == 5: # VP  ODE
                score = - eps / self.sqrtmab[i]
                x_i = (2 - torch.sqrt(1-self.beta_t[i])) * x_i + 0.5 * self.beta_t[i] * score
        return x_i

    def sample_score(self, x_i, t_k) -> torch.Tensor:
        with torch.no_grad():
            ts_ = torch.tensor(t_k).to(x_i.device) / self.n_T
            n_sample = z.shape[0]
            ts_ = ts_.repeat(n_sample)
            eps = self.eps_model(x_i,ts_)
            score = - eps / self.sqrtmab[i]
        return score
    def em_sampler(self,n_sample, size, device='cuda'):
        t = torch.ones(n_sample, device=device) # initial t = 1
        init_x = torch.randn(n_sample, *size).to(device)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MLPSkipNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.time_embed_dim = 64
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim,latent_dim)
        )
        # MLP layers
        self.activation = 'silu'
        use_norm = True
        num_layers =20
        num_hid_channels = 2048 # latent_dim * 4
        num_channels = latent_dim
        condition_bias=1
        dropout = 0
        self.skip_layers = list(range(1, num_layers))
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = self.activation
                norm = use_norm
                cond = True
                a, b = num_channels, num_hid_channels
                dropout = dropout
            elif i == num_layers - 1:
                act = 'none'
                norm = False
                cond = False
                a, b = num_hid_channels, num_channels
                dropout = dropout
            else:
                act = self.activation
                norm = use_norm
                cond = True
                a, b = num_hid_channels, num_hid_channels
                dropout = dropout

            if i in self.skip_layers:
                a += num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels,
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout,
                ))
        self.last_act = nn.Identity()

    def forward(self, x, t, z_sem=None):
        
        t = timestep_embedding(t, self.time_embed_dim).to(x.dtype)
        cond = self.time_embed(t)
        h = x
        if z_sem is not None:
            cond += z_sem
        
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return h

class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: str,
        use_cond: bool,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.condition_bias = condition_bias
        self.use_cond = use_cond
        self.activation = activation
        self.linear = nn.Linear(in_channels, out_channels)
        if activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'silu':
                    nn.init.kaiming_normal_(module.weight,
                                             a=0,
                                            nonlinearity='relu')
                else:
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x



class RelsoVAE(BaseModel):
    def __init__(self, hparams):
        super(RelsoVAE, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()
        self.joint_regressor = hparams.joint_regressor

        # model atributes
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim  # nhid
        self.embedding_dim = hparams.embedding_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.nhead = 4
        self.src_mask = None
        self.bz = hparams.batch_size

        self.lr = hparams.lr
        self.model_name = "relso1"

        self.alpha = hparams.alpha_val
        self.gamma = hparams.gamma_val

        self.sigma = hparams.sigma_val

        self.kl_weight = hparams.kl_weight

        try:
            self.eta = hparams.eta_val
        except:
            self.eta = 1.0

        try:
            self.interp_samping = hparams.interp_samp
        except:
            self.interp_samping = True

        try:
            self.negative_sampling = hparams.neg_samp
        except:
            self.negative_sampling = True

        try:
            self.neg_size = hparams.neg_size
            self.neg_floor = hparams.neg_floor
            self.neg_weight = hparams.neg_weight
            self.neg_focus = hparams.neg_focus
            self.neg_norm = hparams.neg_norm
        except:
            self.neg_focus = False

        self.dyn_neg_bool = False  # set False as default

        try:
            self.interp_size = hparams.interp_size
            self.interp_weight = hparams.interp_weight

        except:
            pass
        self.interp_inds = None
        self.dyn_interp_bool = False

        try:
            self.wl2norm = hparams.wl2norm
        except:
            self.wl2norm = False

        self.g_opt_step = 0

        self.seq_len = hparams.seq_len

        # The embedding input dimension may need to be changed depending on the size of our dictionary
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)

        self.pos_encoder = PositionalEncoding(
            d_model=self.embedding_dim, max_len=self.seq_len
        )

        self.glob_attn_module = nn.Sequential(
            nn.Linear(self.embedding_dim, 1), nn.Softmax(dim=1)
        )

        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )
        # make decoder)
        self._build_decoder(hparams)

        # for los and gradient checking
        self.z_rep = None

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.embedding_dim, self.latent_dim)
        aux_params = {"latent_dim": self.latent_dim, "probs": hparams.probs}
        aux_hparams = argparse.Namespace(**aux_params)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            self.regressor_module = auxnetwork(aux_hparams)
        except:
            print('aux network loading failed - defaulting to SpectralRegressor')
            auxnetwork = str2auxnetwork("spectral")
            self.regressor_module = auxnetwork(aux_hparams)

    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _build_decoder(self, hparams):
        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            Block(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=3, padding=1),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

        print(dec_layers)

    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            )
        # self.embed gives output (batch_size,sequence_length,num_features)
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)

        return output_embed

    def encode(self, batch):
        embedded_batch = self.embed(batch)
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed)  # output should be B x S x 1
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        mu, logvar = self.bottleneck_module(z_rep)
        logvar = torch.log(torch.ones_like(logvar) * 0.008)

        return mu, logvar

    def decode(self, z_rep):

        h_rep = z_rep  # B x 1 X L

        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)

            h_rep = layer(h_rep)

        return h_rep

    def predict_fitness(self, batch):

        z = self.encode(batch)

        y_hat = self.regressor_module(z)

        return y_hat

    def predict_fitness_from_z(self, z_rep):

        # to regain the batch dimension
        if len(z_rep) == 1:
            z_rep = z_rep.unsqueeze(0)

        y_hat = self.regressor_module(z_rep)

        return y_hat

    def interpolation_sampling(self, z_rep):
        """
        get interpolations between z_reps in batch
        interpolations between z_i and its nearest neighbor
        Args:
            z_rep ([type]): [description]
        Returns:
            [type]: [description]
        """
        z_dist_mat = self.pairwise_l2(z_rep, z_rep)
        k_val = min(len(z_rep), 2)
        _, z_nn_inds = z_dist_mat.topk(k_val, largest=False)
        z_nn_inds = z_nn_inds[:, 1]

        z_nn = z_rep[z_nn_inds]
        z_interp = (z_rep + z_nn) / 2

        # use 10 % of batch size
        subset_inds = torch.randperm(len(z_rep), device=self.device)[: self.interp_size]

        sub_z_interp = z_interp[subset_inds]
        sub_nn_inds = z_nn_inds[subset_inds]

        self.interp_inds = torch.cat(
            (subset_inds.unsqueeze(1), sub_nn_inds.unsqueeze(1)), dim=1
        )

        return sub_z_interp

    def add_negative_samples(
        self,
    ):

        max2norm = torch.norm(self.z_rep, p=2, dim=1).max()
        wandb.log({"max2norm": max2norm})
        rand_inds = torch.randperm(len(self.z_rep))
        if self.neg_focus:
            neg_z = (
                0.5 * torch.randn_like(self.z_rep)[: self.neg_size]
                + self.z_rep[rand_inds][: self.neg_size]
            )
            neg_z = neg_z / torch.norm(neg_z, 2, dim=1).reshape(-1, 1)
            neg_z = neg_z * (max2norm * self.neg_norm)

        else:
            center = self.z_rep.mean(0, keepdims=True)
            dist_set = self.z_rep - center

            # gets maximally distant rep from center
            dist_sort = torch.norm(dist_set, 2, dim=1).reshape(-1, 1).sort().indices[-1]
            max_dist = dist_set[dist_sort]
            adj_dist = self.neg_norm * max_dist.repeat(len(self.z_rep), 1) - dist_set
            neg_z = self.z_rep + adj_dist
            neg_z = neg_z[rand_inds][: self.neg_size]


        return neg_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, batch):

        mu, logvar = self.encode(batch)
        z_rep = self.reparameterize(mu, logvar)
        self.z_rep = z_rep

        x_hat = self.decode(z_rep)

        y_hat = self.regressor_module(z_rep)

        # return [x_hat, y_hat], z_rep
        return [x_hat, z_rep, mu, logvar, y_hat], z_rep

    def pairwise_l2(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, 2).sum(2)

        return dist

    def loss_function(self, predictions, targets, valid_step=False):

        # unpack everything
        x_hat, z_rep, mu, logvar, y_hat = predictions
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        x_true, y_true = targets

        recon_x = x_hat

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(22, device=self.device)
        ce_loss_weights[21] = 0.8

        vae_loss = nn.CrossEntropyLoss(reduction='none', weight=ce_loss_weights)(recon_x, x_true)
        vae_loss = vae_loss.mean(axis=1)

        reg_loss = nn.MSELoss()(y_hat.flatten(), y_true.flatten())

        total_loss = vae_loss.mean() + self.kl_weight * loss_kl.mean()

        if self.joint_regressor:
            total_loss = total_loss + reg_loss
        
        mloss_dict = {
            "ae_loss": vae_loss.mean(),
            "reg_loss": reg_loss,
            "loss": total_loss,
        }

        return total_loss, mloss_dict
