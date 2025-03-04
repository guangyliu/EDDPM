# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
from my_transformers import *
import argparse
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
import torch.utils.data.distributed
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import numpy as np
import torch
import torch.nn.init as init
# from run_latent_generation import sample_sequence_conditional
from nltk.translate.bleu_score import corpus_bleu
from transformers import AdamW  # ,OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import GPT2LMHeadModel as GPT2_
from modules import VAE, DDPM, LinearModel, MLPSkipNet, UNetModel,DenseEmbedder, sample_sequence_conditional, TransformerNet
from utils import (calc_iwnll, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear, BucketingMultipleFiles_DataLoader, MultipleFiles_DataLoader)
import sys
sys.path.append("../..")
from train_ddpm_latent import calc_ppl_lgy_ddpm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pudb.remote import set_trace
import shutil, sys
sys.path.append("examples/big_ae")

def collate(examples):
    # Convert to Tensors and build dataset

    input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples],
                                  batch_first=True, padding_value=bert_pad_token)
    input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples],
                                 batch_first=True, padding_value=gpt2_pad_token)
    try:
        token_lengths = torch.tensor([[len(f['bert_token']), len(f['gpt2_token'])] for f in examples],
                                     dtype=torch.long)
    except:
        token_lengths = torch.zeros((len(examples), 1091))
        for i in range(len(examples)):
            token_lengths[i, len(examples[i]['gpt2_token'])] = 1
    return (input_ids_bert, input_ids_gpt, token_lengths)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': GPT2ForLatentConnector,
    # 'openai-gpt': (None, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bertu': BertForLatentConnectorAVG,
    'bert': BertForLatentConnector,
    'roberta': RobertaForLatentConnector,
    'deberta': DebertaForLatentConnector,
    't5': T5EncoderForLatentConnector,
    'albert':AlbertForLatentConnector,
    'llama':LlamaForLatentConnector
}

parameter_name = []


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args,
                                          file_path=args.eval_data_file if evaluate else args.train_data_file,
                                          block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args,
                                    file_path=args.eval_data_file if evaluate else args.train_data_file,
                                    block_size=args.block_size)
    return dataset


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        file_path = args.train_data_file
        dataloader = MultipleFiles_DataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=True)
    else:
        pass
    if evaluate:
        args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        file_path = args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100,
                                        shuffle=False)
    return dataloader


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
    labels[masked_indices == 1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)


def save_checkpoint(model_vae, optimizer, global_step, args, ppl=False, ddpm=None):
    # Create output directory if needed
    # Save model checkpoint
    save_last = 1
    model_to_save = model_vae.module if hasattr(model_vae,
                                                'module') else model_vae  # Take care of distributed/parallel training
    ddpm_to_save = ddpm.module if hasattr(ddpm, 'module') else ddpm
    state_dict_new = {}
    state_dict = model_to_save.state_dict()
    ddpm_state_dict = ddpm_to_save.state_dict()
    for key in parameter_name:
        if key in state_dict.keys():
            state_dict_new[key] = state_dict[key]

    checkpoint = {
        'iter': global_step,
        'model_state_dict': state_dict_new,  # model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'beta': model_to_save.args.beta,
        'args': args
    }
    checkpoint_ddpm = {
        'iter': global_step,
        'model_state_dict': ddpm_state_dict,
        'args': args
    }
    if ppl:
        save_last = 2
    output_ddpm_dir = os.path.join(args.output_dir, 'checkpoint-ddpm-{}'.format(save_last))
    if not os.path.exists(output_ddpm_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_ddpm_dir)
    torch.save(checkpoint_ddpm, os.path.join(output_ddpm_dir, 'training_ddpm.bin'))
    logger.info("Saving DDPM checkpoint to %s", output_ddpm_dir)
    
    output_full_dir = os.path.join(args.output_dir, 'checkpoint-full-{}'.format(save_last))
    if not os.path.exists(output_full_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_full_dir)

    logger.info("Start saving full model checkpoint to %s", output_full_dir)
    if args.use_philly:
        save_solid = False
        n_save_attempts = 0
        while not save_solid:
            try:
                n_save_attempts += 1
                logger.info(f"Saving full checkpoint: {n_save_attempts} attempts made")
                torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
                logger.info("Saving full checkpoint to %s,", output_full_dir)
                save_solid = True
            except:
                pass
    else:
        torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        logger.info("Saving full checkpoint to %s", output_full_dir)

class VAE_DDPM(nn.Module):
    def __init__(self, model_vae, ddpm) :
        super(VAE_DDPM, self).__init__()
        self.model_vae = model_vae
        self.ddpm = ddpm

    def forward(self,inputs, labels, std=False, return_z=False, return_mu=False): 
        
        loss_rec, loss_kl, loss, latent_z, mu = self.model_vae(inputs, labels, std=std, return_z=return_z, return_mu=return_mu)
        ddpm_loss, loss_weight = self.ddpm.forward_new(latent_z, mu)
        
        if self.model_vae.args.ddpm_weight > 0:
            loss = (1/(loss_weight * self.model_vae.args.nt)  * loss).mean() + self.model_vae.args.ddpm_weight *ddpm_loss.mean()
        else:
            loss = loss.mean() + 0.0* ddpm_loss.mean()
        # loss = (1/(loss_weight * self.model_vae.args.nt)  * loss).mean() + self.model_vae.args.ddpm_weight *ddpm_loss.mean()
        return loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight

def train(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, table_name, eval_dataloader,checkpoint=None, ddpm=None):
    """ Train the model """
    torch.cuda.set_device(args.local_rank)
    torch.cuda.empty_cache()

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/' + args.output_dir.split('/')[-2] + '/' + args.output_dir.split('/')[-1])

    args.train_batch_size = args.per_gpu_train_batch_size
    num_files = 1
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     # args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)

    # model_encoder, model_decoder, model_connector = model_vae.encoder,  model_vae.decoder, model_vae.linear
    model = VAE_DDPM(model_vae, ddpm)
    model = model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    if args.fix_model == 84:
        def condition_f(n):
            return ('linear' in n or 'wte' in n or 'decoder.transformer.h.0' in n or 'encoder' in n)
    
    elif args.fix_model == 841:
        def condition_f(n):
            return ('linear' in n or 'lora' in n or 'encoder' in n)
    para = [p for n, p in model.model_vae.named_parameters() if condition_f(n)]
    if args.ddpm_weight > 0:
        para.extend([p for n, p in model.ddpm.named_parameters()])
    if not args.fp16:
        optimizer_grouped_parameters = [
            {'params': para,
                'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        import apex
        from apex import amp

        optimizer = apex.optimizers.FusedAdam(para, lr=args.learning_rate, eps=args.adam_epsilon)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        # ddpm = amp.initialize(ddpm, opt_level=args.fp16_opt_level)
    parameter_name.extend([n for n, _ in model.model_vae.named_parameters() if
                            condition_f(n)])
    parameter_name.extend([n for n, _ in model.ddpm.named_parameters()])
    from transformers import get_polynomial_decay_schedule_with_warmup
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=t_total, lr_end=5e-7, power=3.0)

    # multi-gpu training (should be after apex fp16 initialization)
    

    # Distributed training (should be after apex fp16 initialization)
    
    # if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          )
    args.gpu = args.local_rank

    # Train!
    # set_trace(term_size=(120,30))
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", train_dataloader.num_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.per_gpu_train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # model = model.module if hasattr(model,
    #                                         'module') else model  # Take care of distributed/parallel training
    
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    first_dataloader_len = len(train_dataloader)
    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    model.eval()
    if args.local_rank==0:
        with torch.no_grad():
            result_new = calc_rec_lgy(model.module.model_vae, encoder_tokenizer, decoder_tokenizer, args, eval_dataloader, ns=100)
            result_new.update(evaluate(args, model.module.model_vae, encoder_tokenizer, decoder_tokenizer, table_name,eval_dataloader))
            for key, value in result_new.items():
                logger.info('eval_%s:%f',key,value)
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            results_new = calc_ppl_lgy_ddpm(
                            model.module.model_vae, encoder_tokenizer, decoder_tokenizer, args, 1,
                            model.module.ddpm, model_ppl, tokenizer_ppl, z=None
                        )
            for key, value in result_new.items():
                logger.info('eval_%s:%f',key,value)
                tb_writer.add_scalar('eval_DDPM_{}'.format(key), value, global_step)
                # logger.info("DDPM_%s:%s",str(key),str(value))
        logger.info('\nBLEU is %f\n"', result_new['bleu'])
        for key, value in results_new.items():
            # logger.info('eval_%s:%f',key,value)
            tb_writer.add_scalar('eval_DDPM_{}'.format(key), value, global_step)
            logger.info("DDPM_%s:%s",str(key),str(value))
        logger.info('\nBLEU is %f\n"', result_new['bleu'])
        torch.cuda.empty_cache()
    torch.distributed.barrier()
    # beta_t_list = frange_cycle_zero_linear(n_iter, start=5.0, stop=args.ddpm_weight, n_cycle=1,
    #                                     ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    best_bleu = 0
    best_ppl = 100
    dtype_ = torch.half if args.fp16 else torch.float
    if args.logging_steps == -1:
        args.logging_steps = len(train_dataloader) if len(train_dataloader)<2500 else 2500
    pbar_update = 100 if args.logging_steps > 1000 else args.logging_steps //5
    for epoch in train_iterator:
        # train_dataloader.reset()
        model.zero_grad()
        for idx_file in range(1):

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False) 

            args.save_steps = args.logging_steps
            for step, batch in enumerate(epoch_iterator):
                # set_trace(term_size=(120,30))
                tokenized_text0, tokenized_text1, _ = batch
                inputs, labels = tokenized_text0, tokenized_text1
                labels = tokenized_text1

                tokenized_text1 = tokenized_text1.to(args.device)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                model.module.model_vae.args.fb_mode = 5 #  
                model.train()
                loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight = model(inputs, labels, return_z=True)
                # ddpm_loss = ddpm.forward_ours(latent_z, mu, logvar)
                # ddpm_loss, loss_weight = ddpm.forward_new(latent_z.to(dtype_), mu.to(dtype_)) # 64 , loss_weight
                if train_step % 100 == 0:
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('loss_rec_train', loss_rec.mean().item(), train_step)
                        tb_writer.add_scalar('loss_kl_train', loss_kl.mean().item(), train_step)
                        tb_writer.add_scalar('loss_train', loss.mean().item(), train_step)
                        tb_writer.add_scalar('lr_train', scheduler.get_last_lr()[0], train_step)
                        tb_writer.add_scalar('loss_ddpm_train', ddpm_loss.mean().item(), train_step)
                    torch.distributed.barrier()
                train_step += 1
                loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
                loss_kl = loss_kl.mean()
                
                # if not args.use_pretrained_vae and epoch == 0 and step < 2000 :
                #     loss = (1/(loss_weight * args.nt)  * loss).mean() + 1 * ddpm_loss.mean()
                # else:
                # loss = (1/(loss_weight * args.nt)  * loss).mean() + args.ddpm_weight *ddpm_loss.mean()
                # logging_step_ = (args.logging_steps // 100) if args.logging_steps> 100 else logging_step_// 5
                if train_step % pbar_update == 0:
                    epoch_iterator.set_description(
                        (
                            f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                            f'loss_rec: {loss_rec.item():.3f}; ddpm: {ddpm_loss.mean().item():.3f}; '
                        )
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()  # Update learning rate schedule

                    model.zero_grad()
                    
                    global_step += 1

                    if args.local_rank in [0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.local_rank == 0 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            #                         args.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size // 2
                            model.eval()
                            with torch.no_grad():
                                results = evaluate(args, model.module.model_vae, encoder_tokenizer, decoder_tokenizer, table_name,eval_dataloader)
                                #                         args.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size * 2
                                results_new = calc_ppl_lgy_ddpm(
                                                model.module.model_vae, encoder_tokenizer, decoder_tokenizer, args, 1,
                                                model.module.ddpm, model_ppl, tokenizer_ppl, z=None
                                            )
                                for key, value in results_new.items():
                                    logger.info("DDPM_"+key+": %s",str(results_new[key]))
                                    tb_writer.add_scalar('eval_{}'.format("DDPM_"+key), value, global_step)
                                results.update(calc_rec_lgy(model.module.model_vae, encoder_tokenizer, decoder_tokenizer, args, eval_dataloader,ns=100))
                            #                         results['ppl_sample'] = sampling_lgy(model_vae, decoder_tokenizer, args, LM_model, LM_tokenizer)['ppl']
                            #                         results['bleu'] = result_new['bleu']
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if results['bleu'] >= best_bleu:
                            best_bleu = results['bleu']
                            if not args.no_save:
                                save_checkpoint(model.module.model_vae, optimizer, global_step, args,ddpm=model.module.ddpm)
                        if 12 < results_new['ppl'] < best_ppl and results_new['norm_z'] < 12 and global_step > 2 * args.logging_steps:
                            best_ppl = results_new['ppl']
                            if not args.no_save:
                                tb_writer.add_scalar('eval_best_ppl', best_ppl, global_step)
                                tb_writer.add_scalar('eval_best_bleu', results['bleu'], global_step)
                                save_checkpoint(model.module.model_vae, optimizer, global_step, args,ppl=True, ddpm=model.module.ddpm)
                        logger.info("Current Path is %s", args.output_dir)
                    torch.distributed.barrier()

                # if args.max_steps > 0 and global_step > args.max_steps:
                #     epoch_iterator.close()
                #     break

        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break
    results = calc_rec_lgy(model.module.model_vae, encoder_tokenizer, decoder_tokenizer, args, eval_dataloader,ns=100)
    #     results['ppl_sample'] = sampling_lgy(model_vae, decoder_tokenizer, args, LM_model, LM_tokenizer, cnt=1000)['ppl']
    # for key, value in results.items():
    #     tb_writer.add_scalar('final_{}'.format(key), value, global_step)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer



def evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer, table_name,eval_dataloader, prefix="", subset="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_vae.eval()

    model_vae = model_vae.module if hasattr(model_vae,
                                            'module') else model_vae  # Take care of distributed/parallel training
    # mi = calc_mi(model_vae, eval_dataloader, args)
    mi = 0
    au = calc_au(model_vae, eval_dataloader, delta=0.01, args=args)[0]
    # ppl, elbo, nll, kl = calc_iwnll(model_vae, eval_dataloader, args, ns=100)
    ppl, elbo, nll, kl = 0,0,0,0
    result = {
        "perplexity": ppl, "elbo": elbo, "kl": kl, "nll": nll, "au": au, "mi": mi
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    row = {
        'PartitionKey': 'MILU_Rule_Rule_Template',
        'RowKey': str(datetime.now()),
        'ExpName': args.ExpName,
        'test_perplexity': str(ppl),
        'test_elbo': str(elbo),
        'test_nll': str(nll),
        'test_au': str(au),
        # 'test_mi': str(mi)
    }
    # pdb.set_trace()
    # ts.insert_entity(table_name, row)

    return result


def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, eval_dataloader,ns=1):
    from modules import sample_sequence_conditional
    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    result = defaultdict(str)
    ref = []
    cand = []
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction", disable=args.disable_bar):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)
        context_tokens = decoder_tokenizer.encode(decoder_tokenizer.bos_token)
        with torch.no_grad():
            # text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)
            attention_mask = (x0 != encoder_tokenizer.pad_token_id).float()

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask)[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            # latent_z = model_vae.reparameterize(mean, logvar, nsamples=1).squeeze(1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x_lengths[0, 1],  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )

        for i in range(latent_z.size(0)):
            text_x0_ = decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x0_ = text_x0_.split()
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x1 = text_x1.split()

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)

        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    logger.info("  BLEU = %s", str(round(bleu, 2)))
    output_eval_file = os.path.join(args.output_dir, "eval_results_bleu.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_eval_file, "w") as writer:
        writer.write("%s = %s\n" % ('bleu', str(bleu)))
    return {'bleu': bleu}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", action='store_true',
                        help="Use sentence length re-weight the reconstruction loss.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--latent_as_gpt_memory", default=1, type=int,
                        help="Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", default=1, type=int, help="Latent vector as embeddings for GPT2.")

    ## Objective functions
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="The weighting hyper-parameter of the KL term in VAE")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")
    parser.add_argument("--block_size", default=30, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.1, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.5, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=1, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=int, default=-1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    # Precision & Distributed Training 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    ############## Mine
    parser.add_argument('--fix_model', type=int, default=84,
                        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers")
    parser.add_argument('--disable_bar', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--nt', type=int, default=1000, help="T for diffusion process")
    parser.add_argument('--shell_name', type=str, default='', help="shell name")
    parser.add_argument("--ddpm_pretrain", type=int, default=0,
                        help="Use pretrained DDPM")
    parser.add_argument('--ddpm_weight', type=float, default=1.0)
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    # backup codes
    if True:
        code_bk_path = os.path.join(args.output_dir,"code_bk")
        try:
            if not os.path.exists(code_bk_path): 
                os.makedirs(code_bk_path)
        except:
            print('skip create folder')
        current_file = sys.argv[0]
        file_list = [current_file, 'examples/big_ae/modules/vae.py',args.shell_name]
        for file in file_list:
            shutil.copy(file,code_bk_path)
        print('backup the codes to: '+code_bk_path)
    model_id ='gpt2'
    if args.dataset == 'yelp':
        model_id =  '../classifiers/gpt2_yelp'
    else:
        print('not implemented')
    # # + args.output_dir.split('/')[-1]  # sentiment'  # _sentiment' #amazon'
    print(model_id)
    global model_ppl
    model_ppl = GPT2_.from_pretrained(model_id,local_files_only=False).cuda()
    global tokenizer_ppl
    tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id,local_files_only=False)
    

    if args.fix_model == 3 or args.fix_model == 4:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
        MODEL_CLASSES['bert'] = BertForLatentConnectorNew
        MODEL_CLASSES['roberta'] = RobertaForLatentConnectorNew
        MODEL_CLASSES['deberta'] = DebertaForLatentConnectorNew
    elif args.fix_model == 5:
        # gpt2 unchanged
        MODEL_CLASSES['bert'] = BertForLatentConnectorNew
        MODEL_CLASSES['roberta'] = RobertaForLatentConnectorNew
        MODEL_CLASSES['deberta'] = DebertaForLatentConnectorNew
    elif args.fix_model == 6 or args.fix_model == 8 or args.fix_model == 8 or args.fix_model == 83 or args.fix_model == 881  or args.fix_model == 882 or args.fix_model == 883:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew#
    elif args.fix_model == 84:
        MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
        if 'large' or 'xl' in args.decoder_model_name_or_path:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
        else:
            MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2
    elif args.fix_model == 85:
        MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
    elif args.fix_model == 13 or args.fix_model == 14 or args.fix_model == 82:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    # if args.decoder_model_type in ["bert", "roberta"] and not args.mlm:
    #     raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
    #                      "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = torch.cuda.device_count()

    # args.n_gpu = torch.distributed.get_world_size()
    args.world_size = torch.distributed.get_world_size()
    args.device = device
    # set_trace(term_size=(120,30))
    # Setup logging
    import time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,)
                        # filename=os.path.join(args.output_dir,'logging'+time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())+'.out'))
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.ExpName = 'Vae_' + args.dataset + '_Nz_' + str(args.latent_size) + '_Beta_' + str(args.beta) + '_Dkl_' + str(
        args.dim_target_kl) + '_Ra_' + str(args.ratio_increase) + '_R0_' + str(args.ratio_zero)
    table_name = 'Vae' + args.dataset + 'Nz' + str(args.latent_size)
    try:
        ts.create_table(table_name)
    except:
        pass

    # Set seed
    set_seed(args)
    # if 'roberta' in args.encoder_model_type:
    #     print("This is ROBERTA, block size modified")
    #     args.block_size = args.block_size + 1
    # if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # Load Optimius pre-trained model and tokenizer
    checkpoint = None
    if args.use_pretrained_model:
        global_step = args.gloabl_step_eval
        output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))
        checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'),map_location=torch.device('cuda', args.local_rank))
        if args.ddpm_pretrain:
            ddpm_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-ddpm-{}'.format(global_step))
            ddpm_checkpoint = torch.load(os.path.join(ddpm_full_dir, 'training_ddpm.bin'),map_location=torch.device('cuda', args.local_rank))
    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
    # encoder_config = encoder_config_class.from_pretrained(
    #     args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)
    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case, local_files_only=False)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id,local_files_only=False)

    # model_encoder.to(args.device)

    ## Decoder
    decoder_model_class = MODEL_CLASSES[args.decoder_model_type]
    if 'llama' not in args.decoder_model_name_or_path:
        tokenizer_decoder = AutoTokenizer.from_pretrained(
            args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
            do_lower_case=args.do_lower_case, local_files_only=False)
    else:
        from transformers import LlamaTokenizer
        tokenizer_decoder = LlamaTokenizer.from_pretrained(
            args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
            do_lower_case=args.do_lower_case, local_files_only=False, bos_token='<s>', eos_token='</s>',pad_token='</s>')
        
        
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    if args.latent_as_gpt_emb + args.latent_as_gpt_memory == 0:
        return  # latent vector should pass into GPT to decode
    else:
        latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
        latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

    # setattr(decoder_config, "latent_size", args.latent_size)
    
    if 'llama' not in args.decoder_model_name_or_path:
        
        model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory,local_files_only=False)
        decoder_n_layer = model_decoder.transformer.config.n_layer
    else:
        print('load LLAMA model')
        model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory,local_files_only=False,
                                                        ) # torch_dtype=torch.float16
        decoder_n_layer = model_decoder.transformer.config.num_hidden_layers

    
    if args.fix_model == 3 or args.fix_model == 4:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[-1].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
        print('Change the Order of Decoder Layers')
    elif args.fix_model == 5:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[0].state_dict())
    elif args.fix_model == 6 or args.fix_model == 8 or args.fix_model == 85 or args.fix_model == 881  or args.fix_model == 882 or args.fix_model == 883:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
    elif args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order() # GPT-2-XL 1.5B  LORA, 
    elif args.fix_model == 10 or args.fix_model == 11:
        from transformers.adapters import CompacterConfig
        config = CompacterConfig(reduction_factor=4)
        model_decoder.transformer.add_adapter("dummy", config=config)
        model_decoder.transformer.train_adapter("dummy")
        # model_decoder.transformer.train_adapter("poem")
    elif args.fix_model == 841:
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05, bias="none",target_modules=['c_attn','c_proj','c_fc']#target_modules=['q_proj','k_proj','v_proj','o_proj'],
        ) # 8 16 
        model_decoder = get_peft_model(model_decoder, peft_config)
 
    elif args.fix_model == 13 or args.fix_model == 14 or args.fix_model == 82:
        model_decoder.transformer.h[decoder_n_layer+1].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.change_order(extra_num=2)
    elif args.fix_model == 83:
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.config.n_layer += 1
    # Save the init weights of BERT and GPT-2, so that we can load from local (Some infra requires so)
    if args.save_bert_gpt_init:
        encoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-enoder-{args.latent_size}")
        if not os.path.exists(encoder_path): os.makedirs(encoder_path)
        model_encoder.save_pretrained(encoder_path)
        tokenizer_encoder.save_pretrained(encoder_path)

        decoder_path = os.path.join(args.output_dir, f"initial-models-tokenization-decoder-{args.latent_size}")
        if not os.path.exists(decoder_path): os.makedirs(decoder_path)
        model_decoder.save_pretrained(decoder_path)
        tokenizer_decoder.save_pretrained(decoder_path)


    if 'llama' not in args.decoder_model_name_or_path:
        # Chunyuan: Add Padding token to GPT2
        special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
        num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
        # import ipdb
        # ipdb.set_trace()
        print('We have added', num_added_toks, 'tokens to GPT2')
        model_decoder.resize_token_embeddings(
        len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        assert tokenizer_decoder.pad_token == '<PAD>'
    global bert_pad_token, gpt2_pad_token
    bert_pad_token = tokenizer_encoder.pad_token_id
    gpt2_pad_token = tokenizer_decoder.pad_token_id

    # model_decoder.to(args.device)

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)

    # pdb.set_trace()
    if args.use_random_weight:
        model_vae.apply(weights_init_rondom)

    if args.use_pretrained_model:
        if args.fix_model == 4 or args.fix_model == 6:
            key_list = [n for n in checkpoint['model_state_dict']]
            for key in key_list:
                if 'linear' in key:
                    checkpoint['model_state_dict'].pop(key)
                    print('drop', key)

        if args.fix_model == 7:
            key_list = [n for n in checkpoint['model_state_dict']]
            for key in key_list:
                if 'linear' in key:
                    checkpoint['model_state_dict'].pop(key)
                    print('drop', key)

        model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #
    ddpm = DDPM(eps_model=MLPSkipNet(args.latent_size), betas=(1e-4, 0.02), n_T=args.nt, criterion=nn.MSELoss(reduction='none'),)
    # ddpm = DDPM(eps_model=TransformerNet((args.latent_size)),  betas=(1e-4, 0.02), n_T=args.nt, criterion=nn.MSELoss(reduction='none'),)
    ddpm.apply(weights_init_rondom)
    if args.ddpm_pretrain and args.use_pretrained_model:
        ddpm.load_state_dict(ddpm_checkpoint['model_state_dict'], strict=False)
    ddpm.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    ##############################
    # Training
    global_step = 0
    if args.do_train:
        #store and load from cache in the future
        train_eval_datasets=load_dataset(args.train_data_file)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_eval_datasets['train'])
        
        train_dataloader =  DataLoader(train_eval_datasets['train'], num_workers=0,  pin_memory=True, sampler=train_sampler,collate_fn=collate,batch_size=args.per_gpu_train_batch_size)
        eval_dataloader =  DataLoader(train_eval_datasets['test'], num_workers=0, collate_fn=collate,batch_size=args.per_gpu_eval_batch_size)
        
        
        # train_dataloader = None
        # if args.local_rank == 0:
        #     torch.distributed.barrier()

        global_step, tr_loss, optimizer = train(args, train_dataloader, model_vae, tokenizer_encoder, tokenizer_decoder,
                                                table_name, eval_dataloader,checkpoint, ddpm=ddpm)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

 

    return None


if __name__ == "__main__":
    main()
