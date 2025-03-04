# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
# from ..run_latent_generation import sample_sequence_conditional

# ----------------------------------------------------------------------------
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).to(torch.uint8)
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


def sample_sequence_conditional_sample(model, length, context, past=None, num_samples=1, temperature=1, top_k=0,
                                       top_p=0.0,
                                       device='cpu', decoder_tokenizer=None, eos_id=50259, loss=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    bz = context.size(0)
    with torch.no_grad():
        for ii in range(length):

            inputs = {'input_ids': generated, 'past': past}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            next_token_logits = outputs[0][:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            # next_token = F.softmax(next_token_logits, dim=-1).max(-1, keepdim=True)[1]
            generated = torch.cat((generated, next_token), dim=1)
            tmp = (next_token.squeeze() == eos_id)
            if ii == 0:
                tmp22 = torch.zeros_like(tmp, device='cuda')
            tmp22 = torch.logical_or(tmp22, tmp)
            if False not in tmp22:
                break
        if loss:
            outputs = model(input_ids=generated, past=past, labels=generated,
                            label_ignore=decoder_tokenizer.pad_token_id)
            rec_loss = (-outputs[0]).tolist()
            return generated, rec_loss
    return generated


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                                device='cpu', decoder_tokenizer=None, eos_id=50259, loss=False):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    # sets = set()
    bz = context.size(0)
    t1, t2, t3 = 0, 0, 0
    alist = list(range(bz))
    with torch.no_grad():
        for ii in range(length):
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            # next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            next_token = F.softmax(next_token_logits, dim=-1).max(-1, keepdim=True)[1]
            generated = torch.cat((generated, next_token), dim=1)
            tmp = (next_token.squeeze() == eos_id)
            if ii == 0:
                tmp22 = torch.zeros_like(tmp, device='cuda')
            tmp22 = torch.logical_or(tmp22, tmp)
            if False not in tmp22:
                break
        if loss:
            outputs = model(input_ids=generated, past=past, labels=generated,
                            label_ignore=decoder_tokenizer.pad_token_id)
            rec_loss = (-outputs[0]).tolist()
            return generated, rec_loss
    return generated


class Sampling():
    def __init__(self, batch_size, latent_dim, n_classes, ccf, device, save_path):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.ccf = ccf
        self.device = device
        self.save_path = save_path

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        self.plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))



class ConditionalTransfer(Sampling):
    def __init__(self, sampler, batch_size, latent_dim, n_classes, ccf, device, save_path, ode_kwargs,
                 ld_kwargs, sde_kwargs, every_n_plot=5, model_kwargs=None, test_data_batch=None, sampling=False, ddpm=None):
        super().__init__(batch_size, latent_dim, n_classes, ccf, device, save_path)

        self.sampler = partial(sampler, ccf=ccf, device=device, plot=self.plot, every_n_plot=every_n_plot,
                               **ode_kwargs, **ld_kwargs, **sde_kwargs, **model_kwargs)
        self.ccf = ccf
        self.model_vae = model_kwargs['model']
        self.decoder_tokenizer = model_kwargs['dec_tokenizer']
        self.encoder_tokenizer = model_kwargs['enc_tokenizer']
        self.args = model_kwargs['args']
        self.context_tokens = self.decoder_tokenizer.encode('<BOS>')
        # self.z_k = [torch.FloatTensor(batch_size, latent_dim).normal_(0, 1).to(device) for _ in range(1)]
        self.test_data_batch = test_data_batch
        self.sentence_list = []
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sampling = sampling
        self.ddpm = ddpm
        if self.args.repa_num > 0:
            if 'amazon' in self.args.checkpoint_dir:
                print('Amazon BERT')
                self.cls_tokenizer = AutoTokenizer.from_pretrained('../classifiers/amazon')
                self.cls_model = AutoModelForSequenceClassification.from_pretrained(
                    '../classifiers/amazon').cuda().eval()
            else:
                model_ = 'sentiment'
                self.cls_tokenizer = AutoTokenizer.from_pretrained('../classifiers/'+model_) #bert_sentiment')
                self.cls_model = AutoModelForSequenceClassification.from_pretrained(
                    '../classifiers/'+model_).cuda().eval()
        else:
            self.cls_tokenizer = None
            self.cls_model = None
        self.tense_tokenizer = AutoTokenizer.from_pretrained('../classifiers/sentiment')
        self.tense_model = AutoModelForSequenceClassification.from_pretrained(
            '../classifiers/sentiment').cuda().eval()
        if len(self.args.cls_step.split(',')) >1 :
            self.tense_tokenizer = AutoTokenizer.from_pretrained('../classifiers/tense')
            self.tense_model = AutoModelForSequenceClassification.from_pretrained(
                '../classifiers/tense').cuda().eval()
            if len(self.args.cls_step.split(',')) >2 :
                self.formal_tokenizer = AutoTokenizer.from_pretrained('../classifiers/formality')
                self.formal_model = AutoModelForSequenceClassification.from_pretrained(
                    '../classifiers/formality').cuda().eval()
        # self.z0 = np.zeros((0, 64))
        # self.z1 = np.zeros((0, 64))
        # self.z2 = np.zeros((0, 64))

    def text2latent(self, text):
        # tokenized_text0 = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(text))
        # tokenized_text0 = self.encoder_tokenizer.add_special_tokens_single_sentence(tokenized_text0)
        tokenized_text0 = self.encoder_tokenizer.encode(text)
        inputs_new = torch.tensor([tokenized_text0]).to(self.device)
        bert_fea = self.model_vae.encoder(inputs_new)[1]
        mu, _ = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
        return mu

    def latent2text(self, new_z_k, return_list=False):
        text_list = []
        out,loss = sample_sequence_conditional(
            model=self.model_vae.decoder,
            context=self.context_tokens,
            past=new_z_k.detach(),
            length=30,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            num_samples=new_z_k.shape[0],
            device=self.device,
            decoder_tokenizer=self.decoder_tokenizer,
            eos_id=self.model_vae.eos_token_id,
            loss= True
        )
        text_all = self.decoder_tokenizer.batch_decode(out,clean_up_tokenization_spaces=False, skip_special_tokens=True)
        for i in range(new_z_k.size(0)):
            text_x1 = text_all[i].strip()
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            # print(text_x1)
            text_list.append(text_x1)
        if return_list:
            return text_list
        return text_x1
        # out = sample_sequence_conditional_sample(
        #     model=self.model_vae.decoder,
        #     context=self.context_tokens,
        #     past=new_z_k.detach(),
        #     length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
        #     num_samples=new_z_k.size(0),
        #     device=self.device,
        #     decoder_tokenizer=self.decoder_tokenizer,
        #     eos_id=self.model_vae.eos_token_id
        # )
        # text_x2 = self.decoder_tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=False).split(
        #     '<EOS>')[0].replace('<BOS>', '').strip()
        # text_x2 = text_x2.split()
        # text_x2 = ' '.join(text_x2)
        # print(text_x2)

        # return text_x1,text_x2

    def disturb(self, z, noise_level=0.5, num=5):
        print('ori:\n' + self.latent2text(z) + '\n')
        for i in range(num):
            noise = torch.FloatTensor(1, 64).normal_(0, 0.5).cuda()
            new_z = z + noise
            print('cos:', round(self.cos(new_z, z)[0].item(), 2))
            print('disturb:\n' + self.latent2text(new_z) + '\n')

    def drop_ana(self, text, num_sample=10):
        z_init = self.text2latent(text)
        text_init = self.latent2text(z_init)
        print('ori, cos=1.00:\t', text_init)
        self.model_vae.encoder.train()
        self.model_vae.decoder.eval()
        print('Dropout rate 10%:')
        for jj in range(num_sample):
            z_tmp = self.text2latent(text)
            text_tmp = self.latent2text(z_tmp)
            cos_sim = self.cos(z_init, z_tmp).mean().item()
            print('id' + str(jj) + ', cos=' + str(round(cos_sim, 2)) + ':\t', text_tmp)
        self.model_vae.encoder.eval()

    def repara(self, mu__, logvar__, idx=0, num=5, ori_text=None):
        bleu_list = []
        mu_ = mu__[[idx]]
        logvar_ = logvar__[[idx]]
        # ori = self.sentence_list[idx]
        # print('ori:')
        transfer = self.latent2text(mu_)
        if ori_text:
            ori_split = ori_text.split()
            print(round(sentence_bleu([ori_split], transfer.split(), smoothing_function=SmoothingFunction().method1) * 100, 2))
        # ori_split = [ori.split()]
        # transfer_split = transfer.split()
        # bleu_list.append(
        #     round(sentence_bleu(ori_split, transfer_split, smoothing_function=SmoothingFunction().method1) * 100, 2))
        # print('ori:\n' + ori)
        # print('transfer:\n' + transfer)
        for i in range(num):
            new_z = self.model_vae.reparameterize(mu_, logvar_, 1)[0]
            # print(i)
            transfer = self.latent2text(new_z)
            if ori_text:
                print(round(
                    sentence_bleu([ori_split], transfer.split(), smoothing_function=SmoothingFunction().method1) * 100,
                    2))
            # bleu_list.append(
            #     round(sentence_bleu(ori_split, new_sent.split(), smoothing_function=SmoothingFunction().method1) * 100,
            #           2))
            # print('cos:',round(self.cos(new_z,mu_)[0].item(),2))
            # print(i + 1, new_sent)
        # for i in range(len(bleu_list)):
        #     print(i, '\tbleu:\t', bleu_list[i])

    def _sample_batch(self, sampler, y, z_k, f):
        start_sample_time = time.time()
        if sampler:
            z = sampler(y=y, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z

        else:
            z = z_k.clone()
        ### Debug Begin
        # diff = z - z_k
        # new_z = z - z_k
        # num_samples = z.size(0)
        # similarity = torch.zeros((0),device=z.device)
        # norm = torch.zeros((0), device=z.device)
        # for i in range(num_samples):
        #     similarity = torch.cat((similarity,self.cos(new_z[[i]],new_z[i+1:])))
        #     norm = torch.cat((norm, torch.norm(new_z[[i]]-new_z[i + 1:],dim=-1)))
        # np_norm = norm.detach().cpu().numpy()
        # np_sim = similarity.detach().cpu().numpy()

        # sim = self.cos(z,z_k)
        # norm = torch.norm(z-z_k,dim=1)
        # np_sim=sim.detach().cpu().numpy()
        # np_norm=norm.detach().cpu().numpy()
        # with open('/home/guangyiliu/sim.txt','w') as f:
        #     for i in range(np_sim.shape[0]):
        #         f.write(str(np_sim[i])+'\n')
        # with open('/home/guangyiliu/norm.txt','w') as f:
        #     for i in range(np_sim.shape[0]):
        #         f.write(str(np_norm[i])+'\n')
        # import pdb
        # pdb.set_trace()
        #### vec operation
        # k=2.5
        # z_neg = z_k[:500]
        # z_pos = z_k[500:]
        # diff_vec = z_neg.mean(dim=0) - z_pos.mean(dim=0)
        # neg2pos = z_neg - k*diff_vec
        # pos2neg = z_pos + k*diff_vec
        # zz = torch.cat((neg2pos,pos2neg),0)
        # ### Debug End
        valid_lines = []

        # logvar = -3 * torch.ones_like(z)
        # for ii in range(4):
        #     z = zz[ii*250 : (ii+1)*250]
        #### vec operation
        out = sample_sequence_conditional(
            model=self.model_vae.decoder,
            context=self.context_tokens,
            past=z.detach(),
            length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            num_samples=z.size(0),
            device=self.device,
            decoder_tokenizer=self.decoder_tokenizer,
            eos_id=self.model_vae.eos_token_id
        )
        ############################ begin
        sentent_list = []

        ########################### end
        for i in range(z.size(0)):
            text_x1 = \
                self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                    '<EOS>')[0].replace('<BOS>', '').strip()
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
            if f:
                f.write(text_x1 + '\n')
            ########################## begin
            sentent_list.append(text_x1)
        # repara(z, -5 * logvar)

        ####################### end self.repara(z_k,logvar,0)
        sample_time = time.time() - start_sample_time
        if f:
            return valid_lines, sample_time
        else:
            return sentent_list
    def text2latent_ddpm(self, text):
        # tokenized_text0 = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(text))
        # tokenized_text0 = self.encoder_tokenizer.add_special_tokens_single_sentence(tokenized_text0)
        tokenized_text0 = self.encoder_tokenizer.encode(text)
        inputs_new = torch.tensor([tokenized_text0]).to(self.device)
        bert_fea = self.model_vae.encoder(inputs_new)[1]
        mu, _ = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
        mu = self.ddpm.add_noise(mu)
        return mu

    def latent2text_ddpm(self, new_z_k, loss=False, return_list = False, return_z = False, score_flag=2, T=None):

        new_z_k = self.ddpm.sample_posterior(new_z_k, new_z_k.device, score_flag=score_flag,T=T)
        # logvar = torch.log(torch.zeros_like(new_z_k)+self.ddpm.sigma[0]**2)
        # new_z_k = self.model_vae.reparameterize(new_z_k, logvar,1).squeeze(1)
        # std = self.ddpm.sigma[0]
        # eps = torch.zeros_like(new_z_k).normal_()
        # new_z_k = new_z_k + std * eps
        loss_list = []
        text_list = []
        
        out,loss = sample_sequence_conditional(
            model=self.model_vae.decoder,
            context=self.context_tokens,
            past=new_z_k.detach(),
            length=32,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            num_samples=new_z_k.shape[0],
            device=self.device,
            decoder_tokenizer=self.decoder_tokenizer,
            eos_id=self.model_vae.eos_token_id,
            loss= True
        )
        text_all = self.decoder_tokenizer.batch_decode(out,clean_up_tokenization_spaces=False, skip_special_tokens=True)
        for i in range(new_z_k.size(0)):
            text_x1 = text_all[i].strip()
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            out_loss = round(-np.mean(loss),3)
            loss_list.append(out_loss)
            text_list.append(text_x1)
            # print(out_loss,text_x1)n
        if return_z:
            return text_list,new_z_k
        if return_list :
            return text_list,loss_list
        if loss:
            return text_x1, out_loss
        return text_x1

    def repara_ddpm(self, mu__, logvar__, idx=0, num=5, ori_text=None):
        bleu_list = []
        mu_ = mu__[[idx]]
        logvar_ = logvar__[[idx]]
        # ori = self.sentence_list[idx]
        # print('ori:')
        transfer = self.latent2text_ddpm(mu_)
        if ori_text:
            ori_split = ori_text.split()
            print(round(sentence_bleu([ori_split], transfer.split(), smoothing_function=SmoothingFunction().method1) * 100, 2))
        # ori_split = [ori.split()]
        # transfer_split = transfer.split()
        # bleu_list.append(
        #     round(sentence_bleu(ori_split, transfer_split, smoothing_function=SmoothingFunction().method1) * 100, 2))
        # print('ori:\n' + ori)
        # print('transfer:\n' + transfer)
        new_z = self.model_vae.reparameterize(mu_, logvar_, num).squeeze()
        # print(i)
        transfer = self.latent2text_ddpm(new_z)
        for i in range(num):
            if ori_text:
                print(round(
                    sentence_bleu([ori_split], transfer.split(), smoothing_function=SmoothingFunction().method1) * 100,
                    2))
            # bleu_list.append(
            #     round(sentence_bleu(ori_split, new_sent.split(), smoothing_function=SmoothingFunction().method1) * 100,
            #           2))
            # print('cos:',round(self.cos(new_z,mu_)[0].item(),2))
            # print(i + 1, new_sent)
        # for i in range(len(bleu_list)):
        #     print(i, '\tbleu:\t', bleu_list[i])
    def interpolation(self,z1,z2, num, return_text = False):
        theta = torch.arccos(torch.sum(z1*z2)/(torch.norm(z1)*torch.norm(z2)))
        sin_theta = torch.sin(theta)
        loss_list = []
        text = ''
        for i in range(num):
            j = 1.* i / num
            z_tmp = torch.sin((1-j)*theta)/sin_theta * z1 + torch.sin(j*theta)/sin_theta *z2
            print(i)
            a, loss = self.latent2text_ddpm(z_tmp[None,:],loss=True)
            text+= (str(i)+'\t'+str(round(loss,2))+'\t'+a+'\n')
            loss_list.append(loss)
        if return_text:
            return loss_list, text
        return loss_list
    def linear_interpolation(self,z1,z2, num, return_text=False):

        loss_list = []
        text = ''
        for i in range(num):
            j = 1.* i / num
            z_diff = z2 - z1
            z_tmp = z1 + j * z_diff
            print(i)
            a, loss = self.latent2text_ddpm(z_tmp[None,:],loss=True)
            text+= (str(i)+'\t'+str(round(loss,2))+'\t'+a+'\n')
            loss_list.append(loss)
        if return_text:
            return loss_list, text
        return loss_list
    def interpolation_batch(self,z1,z2, num, return_text = False, T=None, z_kk=None):
        theta = torch.arccos((z1 * z2).sum(1)/(torch.norm(z1,dim=1)*torch.norm(z2,dim=1)))
        sin_theta = torch.sin(theta)
    
        loss_list_all = []
        text = ''
        text_no_del = ''
        text_list_all = []
        ll_list_all = []
        index_list = list(range(num+1)) # [0, 2,4,6,6.5,7,7.4,7.8,8,8.2,8.6,9,9.5,10,12,14, 16]
        # 
        for i in index_list:
            j = 1.* i / num
            tmp1 = torch.matmul(torch.diag(torch.sin((1-j)*theta)/sin_theta) , z1)  
            tmp2 = torch.matmul(torch.diag(torch.sin(j*theta)/sin_theta) ,z2)  
            z_tmp = tmp1 + tmp2 #torch.sin((1-j)*theta)/sin_theta * z1 + torch.sin(j*theta)/sin_theta *z2
            # z_tmp = torch.randn_like(z1)
            print(i)
            likelihood = (-0.5 * (z_tmp.norm(2,dim=1)**2+128*np.log(2*np.pi))).tolist()
            text_list, loss_list = self.latent2text_ddpm(z_tmp,loss=True,return_list=True, T=T)
            text_list_all.append(text_list)  # num * batch
            loss_list_all.append(loss_list)
            ll_list_all.append(likelihood)
            # text+= (str(i)+'\t'+str(round(loss,2))+'\t'+a+'\n')
        text_list_all_T = list(map(list, zip(*text_list_all)))

        for j in range(z_tmp.size(0)):
            for i in range(len(index_list)):
                if text_list_all_T[j][i] not in text_list_all_T[j][:i]:
                    text+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
                text_no_del+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
            text+='\n'
            text_no_del+='\n'
        if return_text:
            return loss_list_all, text, text_no_del
        return loss_list_all
    def linear_interpolation_batch(self,z1,z2, num, return_text = False, T=None):
        # theta = torch.arccos((z1 * z2).sum(1)/(torch.norm(z1,dim=1)*torch.norm(z2,dim=1)))
        # sin_theta = torch.sin(theta)
        loss_list_all = []
        text = ''
        text_no_del = ''
        text_list_all = []
        z_diff = z2 - z1
        index_list = list(range(num+1)) # [0, 2,4,6,6.5,7,7.4,7.8,8,8.2,8.6,9,9.5,10,12,14, 16]
        
        for i in index_list:
            j = 1.* i / num
            z_tmp = z1 + j * z_diff
            print(i)
            text_list, loss_list = self.latent2text_ddpm(z_tmp,loss=True,return_list=True, T=T)
            text_list_all.append(text_list)  # num * batch
            loss_list_all.append(loss_list)
        text_list_all_T = list(map(list, zip(*text_list_all)))

        for j in range(z_tmp.size(0)):
            for i in range(len(index_list)):
                if text_list_all_T[j][i] not in text_list_all_T[j][:i]:
                    text+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
                text_no_del+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
            text+='\n'
            text_no_del+='\n'
        if return_text:
            return loss_list_all, text, text_no_del
        return loss_list_all
    # def linear_interpolation_batch(self,z1,z2, num, return_text = False, T=None):
    #     # theta = torch.arccos((z1 * z2).sum(1)/(torch.norm(z1,dim=1)*torch.norm(z2,dim=1)))
    #     # sin_theta = torch.sin(theta)
    #     loss_list_all = []
    #     text = ''
    #     text_no_del = ''
    #     text_list_all = []
    #     z_diff = z2 - z1
    #     index_list = list(range(1, num+1)) 
    #     all_z = z1.detach()
    #     for i in index_list:
    #         j = 1.* i / num
    #         z_tmp = z1 + j * z_diff
    #         all_z = torch.cat((all_z,z_tmp),dim=0)
            
    #     bz = 500
    #     total_step = int(np.ceil(all_z.size(0)/bz))
    #     for i in range(total_step):
    #         input_z = all_z[i*bz: (i+1)*bz]
    #         text_list, loss_list = self.latent2text_ddpm(input_z,loss=True,return_list=True, T=T)
    #         text_list_all.extend(text_list)  # num * batch
    #         print(i,'out of',total_step)
    #         # loss_list_all.append(loss_list)
    #     text_list_all_new = []
    #     import ipdb
    #     ipdb.set_trace()
    #     for j in range(num+1):
    #         text_t = []
    #         for i in range(z1.size(0)):
    #             text_t.append(text_list_all[*i+j*z1.size(0)])
    #         text_list_all_new.append(text_t)
    #     text_list_all_T = list(map(list, zip(*text_list_all)))



        for j in range(z_tmp.size(0)):
            for i in range(len(index_list)):
                if text_list_all_T[j][i] not in text_list_all_T[j][:i]:
                    text+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
                text_no_del+= (str(index_list[i])+'\t'+text_list_all_T[j][i]+'\n')
            text+='\n'
            text_no_del+='\n'
        if return_text:
            return loss_list_all, text, text_no_del
        return loss_list_all
    
    def interpolation_likelihood(self,z1,z2, num):
        theta = torch.arccos(torch.sum(z1*z2)/(torch.norm(z1)*torch.norm(z2)))
        sin_theta = torch.sin(theta)
        loss_list = []
        for i in range(num):
            j = 1.* i / num
            z_tmp = torch.sin((1-j)*theta)/sin_theta * z1 + torch.sin(j*theta)/sin_theta *z2
            loglikelihood = round(-0.5 * (z_tmp.norm()**2+128*np.log(2*np.pi)).item(),3)
            loss_list.append(loglikelihood)
        return loss_list

    def _sample_batch_repara(self, sampler, y, z_kk, f, repa_num=3):
        start_sample_time = time.time()
        # z_k = self.ddpm.add_noise(z_k)


        # self.interpolation(z_k[0],z_k[2],15)
        ######################
        '''
        SDEidit test code
        '''
        if False:
            def sdedit(step_t):
                with torch.no_grad():
                    z_k = self.ddpm.add_noise_ddpm(z_kk, step_t)
                    z_k_ddim = self.ddpm.add_noise_ddpm(z_kk, step_t)
                    new_z_k = self.ddpm.sample_posterior(z_k, z_k.device, score_flag=1,T=step_t)
                    new_z_k_ddim  =self.ddpm.sample_posterior(z_k_ddim, z_k.device, score_flag=1,T=step_t)
                    text_list = self.latent2text(new_z_k,return_list=True)
                    text_list_ddim = self.latent2text(new_z_k_ddim,return_list=True)
                    ori_list = self.latent2text(z_kk, return_list=True)
                text_list_split = [text.strip().split() for text in text_list]
                text_list_ddim_split = [text.strip().split() for text in text_list_ddim]
                ori_list_split = [[text.strip().split()] for text in ori_list]
                bleu = corpus_bleu(ori_list_split, text_list_split) * 100
                bleu_ddim = corpus_bleu(ori_list_split, text_list_ddim_split) * 100
                print(step_t, bleu, bleu_ddim)
                
                with open('./results/sdedit/'+str(step_t)+'.txt', 'w') as f:
                    f.write('\n'.join(text_list))
                with open('./results/sdedit/'+str(step_t)+'_ddim.txt', 'w') as f:
                    f.write('\n'.join(text_list_ddim))
            sdedit(1)
            sdedit(2)
            sdedit(3)
            sdedit(4)
            sdedit(5)

        '''
        interpolation code
        '''
        if self.args.weight_energy == 100:  # interpolation
            all_step_t = [int(step) for step in self.args.reg_z.split(',')]
            for step_t in all_step_t: #200,500,1000,1500,
                z_k = self.ddpm.add_noise(z_kk, T=step_t) # = z_kk 
                # outtext, outz = self.latent2text_ddpm(z_k, return_z=True, T=step_t)
                # text_list, loss_list = self.latent2text_ddpm(z_kk,loss=True,return_list=True, T=0)
                # ref = [[text.strip().split()] for text in text_list]
                # cand = [text.strip().split() for text in outtext]
                # bleu = corpus_bleu(ref,cand) 
                # z_k = z_kk
                ori_loss_list = []
                batch = z_k.shape[0]
                print('batch',batch)
                all_loss_list = []
                all_linear_list = []
                slerp_text = ''
                slerp_text_no_del = ''
                # lerp_text = ''
                num =50
                int_arr =  []
                from tqdm import trange
                half_num = z_k.shape[0]//2
                print('huaf_num is ', half_num)
                with torch.no_grad():
                    loss_list,slerp_, slerp_no_del= self.interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True, T=step_t, z_kk=z_kk)
                    # loss_list,slerp_, slerp_no_del= self.linear_interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True, T=step_t)
                if len(int_arr) == 0:
                    int_arr = np.array(loss_list).T
                    # int_arr_linear = np.array(linear_list).T
                else:
                    int_arr=np.append(int_arr,np.array(loss_list).T,0)
                    # linear_list = np.append(int_arr_linear,np.array(linear_list).T,0)
                # all_loss_list.extend(loss_list)
                # all_linear_list.extget_samples_multipleend(linear_list)
                slerp_text += (slerp_+'\n')
                slerp_text_no_del+= (slerp_no_del + '\n')
                # lerp_text += (lerp_+'\n')
                # int_arr = np.array(all_loss_list)

                out_arr = np.array([int_arr.mean(0),int_arr.var(0)])

                # int_arr_linear = np.array(all_linear_list)
                # out_linear_arr = np.array([int_arr_linear.mean(0),int_arr_linear.var(0)])
                # with open('int_arr_ll.npy', 'wb') as f:
                #     np.save(f, int_arr)
                with open('./results/interpolation/T_slerp_'+str(step_t)+'old.txt', 'w') as f:
                    # for i in range(out_arr.shape[1]):
                    #     f.write(str(out_arr[0,i])+'\t'+str(out_arr[1,i])+'\n')
                    f.write(slerp_text)
                with open('./results/interpolation/T_slerp_no_del_'+str(step_t)+'old.txt', 'w') as f:
                    # for i in range(out_arr.shape[1]):
                    #     f.write(str(out_arr[0,i])+'\t'+str(out_arr[1,i])+'\n')
                    f.write(slerp_text_no_del)
                # with open('./results/interpolation/out_arr_lerp.txt', 'w') as f:
                #     for i in range(out_linear_arr.shape[1]):
                #         f.write(str(out_linear_arr[0,i])+'\t'+str(out_linear_arr[1,i])+'\n')
                #     f.write(lerp_text)
            import sys
            sys.exit()
        ############
        if False:
            z_k = torch.randn_like(z_k)
            ori_loss_list = []
            batch = z_k.shape[0]
            print('batch',batch)
            all_loss_list = []
            all_linear_list = []
            slerp_text = ''
            # lerp_text = ''
            num =16
            int_arr =  []
            from tqdm import trange
            half_num = z_k.shape[0]//2
            print('huaf_num is ', half_num)
            with torch.no_grad():
                loss_list,slerp_ = self.interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True)
                # linear_list, lerp_ = self.linear_interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True)
            if len(int_arr) == 0:
                int_arr = np.array(loss_list).T
                # int_arr_linear = np.array(linear_list).T
            else:
                int_arr=np.append(int_arr,np.array(loss_list).T,0)
                # linear_list = np.append(int_arr_linear,np.array(linear_list).T,0)
            # all_loss_list.extend(loss_list)
            # all_linear_list.extend(linear_list)
            slerp_text += (slerp_+'\n')
            # lerp_text += (lerp_+'\n')
            # int_arr = np.array(all_loss_list)

            out_arr = np.array([int_arr.mean(0),int_arr.var(0)])

            # int_arr_linear = np.array(all_linear_list)
            # out_linear_arr = np.array([int_arr_linear.mean(0),int_arr_linear.var(0)])
            # with open('int_arr_ll.npy', 'wb') as f:
            #     np.save(f, int_arr)
            with open('./results/interpolation/out_arr_slerp_sample.txt', 'w') as f:
                for i in range(out_arr.shape[1]):
                    f.write(str(out_arr[0,i])+'\t'+str(out_arr[1,i])+'\n')
                f.write(slerp_text)
            # with open('./results/interpolation/out_arr_lerp_sample.txt', 'w') as f:
            #     for i in range(out_linear_arr.shape[1]):
            #         f.write(str(out_linear_arr[0,i])+'\t'+str(out_linear_arr[1,i])+'\n')
            #     f.write(lerp_text)


        #######################
        if sampler:
            z = sampler(y=y, z_k=z_kk.clone())  # 进入 sample_q_ode 函数, 得到new z: bs x latent
        else:
            z = z_kk.clone()
        
        # from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method  
        # import numpy as np
        # with torch.set_grad_enabled(True):
        #     zz=fast_gradient_method(self.ccf.f[0], z_k, eps=0.3,norm=np.inf) #, clip_min=0.01, clip_max=40.)
        #     import pdb
        #     pdb.set_trace()
        #     self.ccf.f[0](zz)
        #     self.latent2text_ddpm(zz[0:1])
        with torch.no_grad():
            logvar = -3 * torch.ones_like(z)
            # self.latent2text_ddpm(z[0:1])
            # print("-")
            # self.repara_ddpm(z,logvar,0)
            # print("0")

            # logvar = logvar.clone()
            new_z = self.model_vae.reparameterize(z, logvar, repa_num)  # bs x num x latent
            ########################### end
            new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
            # new_z = self.ddpm.sample_posterior(new_z.view(-1,self.args.latent_size), new_z.device).view(z.size(0),-1,self.args.latent_size)
            # sentent_list = [[]] * new_z.size(1)
            bleu_list = [0.] * new_z.size(0)
            final_list = [''] * new_z.size(0)
            final_tmp_list = [''] * new_z.size(0)
            bleu_tmp_list = [-0.1] * new_z.size(0)
            idx_list = [0.] * new_z.size(0)

            for ii in range(new_z.size(1)):
                z = new_z[:, ii]
                t1 = time.time()
                if self.sampling:
                    out = sample_sequence_conditional_sample(
                        model=self.model_vae.decoder,
                        context=self.context_tokens,
                        past=z.detach(),
                        length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                        num_samples=z.size(0),
                        device=self.device,
                        decoder_tokenizer=self.decoder_tokenizer,
                        eos_id=self.model_vae.eos_token_id
                    )
                else:
                    out = sample_sequence_conditional(
                        model=self.model_vae.decoder,
                        context=self.context_tokens,
                        past=z.detach(),
                        length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                        num_samples=z.size(0),
                        device=self.device,
                        decoder_tokenizer=self.decoder_tokenizer,
                        eos_id=self.model_vae.eos_token_id
                    )
                t2 = time.time()
                context = torch.tensor(self.context_tokens, dtype=torch.long, device='cuda')
                context = context.unsqueeze(0).repeat(1, 1)

                bleu_tmp = []
                text_tmp = []
                for i in range(z.size(0)):
                    text_x1 = \
                        self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                            '<EOS>')[0].replace('<BOS>', '').strip()
                    text_x1_ = text_x1.split()
                    text_x1 = ' '.join(text_x1_)
                    text_tmp.append(text_x1)
                    bleu = sentence_bleu([self.sentence_list[i].split()], text_x1_,
                                         smoothing_function=SmoothingFunction().method1)
                    bleu_tmp.append(bleu)
                    # cls
                t3 = time.time()
                feature = self.cls_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                logits = self.cls_model(**feature)[0]  # batch x 2
                flag_right = logits.max(-1)[1] == y.squeeze()  # batch == y
                t4 = time.time()
                # print(t2-t1,'\n',t3-t2,'\n',t4-t3)
                num = 0
                for i in range(z.size(0)):
                    if bleu_tmp[i] > bleu_list[i] and flag_right[i]:
                        bleu_list[i] = bleu_tmp[i]
                        final_list[i] = text_tmp[i]
                        idx_list[i] = ii
                    if bleu_tmp[i] > bleu_tmp_list[i]:
                        bleu_tmp_list[i] = bleu_tmp[i]
                        final_tmp_list[i] = text_tmp[i]
                    if ii == new_z.size(1) - 1:  # last one
                        if final_list[i] == '':
                            num += 1
                            final_list[i] = final_tmp_list[i]
                # sentent_list[ii].append(text_x1)
            print(num)
            for i in range(z.size(0)):
                f.write(final_list[i] + '\n')
            #### debug
            # if y[0] == 1:
            #     z_ori = torch.zeros((0, 64), device=z.device)
            #     z_ode = torch.zeros((0, 64), device=z.device)
            #     z_final = torch.zeros((0, 64), device=z.device)
            #     for i in range(z.size(0)):
            #         if idx_list[i] > 0:
            #             zi_final = new_z[[i], idx_list[i]]
            #
            #             zi_ori = z_k[[i]]
            #             zi_ode = z[[i]]
            #             z_final = torch.cat((z_final, zi_final.clone().detach()), 0)
            #             z_ode = torch.cat((z_ode, zi_ode.clone().detach()), 0)
            #             z_ori = torch.cat((z_ori, zi_ori.clone().detach()), 0)
            #     # diff_ode = z_ode - z_ori
            #     # diff_final = z_final - z_ori
            #     # diff_cos = self.cos(diff_ode, diff_final)
            #     # norm_final = torch.norm(diff_final, dim=1)
            #     # norm_ode = torch.norm(diff_ode, dim=1)
            #     self.z0 = np.append(self.z0, z_ori.detach().cpu().numpy(), 0)
            #     self.z1 = np.append(self.z1, z_ode.detach().cpu().numpy(),0)
            #     self.z2 = np.append(self.z2, z_final.detach().cpu().numpy(),0)
            #     zz = torch.cat((z_ori, z_ode, z_final), 0)
            #     npzz = zz.detach().cpu().numpy()
            #     np.save('./npzz.npy',npzz)
            #     # import pdb
            #     # pdb.set_trace()
            #     # ##### debug
            ###################### end
        sample_time = time.time() - start_sample_time
        return None, sample_time
        
    def get_samples_multiple(self,att_val_list='1,1,1', interpolation=False):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)

        repa = self.args.repa_num
########################
        input_list = []
        ori_list= []
        new_list=[]
        cos_list = []
        ori_norm_list = []
        new_norm_list = []
########################
        for att_val in att_val_list.split(';'):
            att_val = [float(i) for i in att_val.split(',')]
            idx = '_'.join(map(str, att_val))
            file_name = os.path.join(self.save_path,
                                     'transfer_multi_' +idx + '.txt')
            f = open(file_name, 'w')
            y = torch.tensor([att_val] * self.batch_size).to(self.device)
            with torch.no_grad():
                for j, batch in enumerate(self.test_data_batch):
                    inputs, x1, tokenized_text_lengths = batch
                    latent_labels = tokenized_text_lengths[:, -1]
                    inputs = inputs.to(self.device)
                    latent_labels = latent_labels.to(self.device)
                    bert_fea = self.model_vae.encoder(inputs, attention_mask=(inputs > 0).float())[1]
                    mu, logvar = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
                    z_k = mu.squeeze()
                    sentent_list = []
                    for i in range(z_k.size(0)):
                        ori_text_x1 = \
                            self.decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                                '<EOS>')[0].replace('<BOS>', '').strip()
                        ori_text_x1 = ori_text_x1.split()
                        ori_text_x1 = ' '.join(ori_text_x1)
                        ########################## begin
                        sentent_list.append(ori_text_x1)
                    self.sentence_list = sentent_list
                    ####################### end
# ######################## 
                    # z_T = self.ddpm.add_noise(z_k)
                    # # text_list_ori, z0_ori = self.latent2text_ddpm(z_T,return_z=True)
                    # text_list_ori = self.latent2text(z_k,return_list=True)
                    # z0_ori = z_k
                    # text_list_new, z0_new = self.latent2text_ddpm(1.3 * z_T,return_z=True)
                    # for tt in range(len(text_list_ori)):
                    #     input_list.append([sentent_list[tt].split()])
                    #     ori_list.append(text_list_ori[tt].split())
                    #     new_list.append(text_list_new[tt].split()) # [['it','is'],['a','v']]
                    # cos_sim_ = (z0_ori * z0_new).sum(1)/(z0_ori.norm(2,dim=1)*z0_new.norm(2,dim=1))
                    # cos_list.extend(cos_sim_.tolist())
                    # ori_norm_list.extend(z0_ori.norm(2,dim=1).tolist())
                    # new_norm_list.extend(z0_new.norm(2,dim=1).tolist())

                    # z_T = z_k
                    # # text_list_ori, z0_ori = self.latent2text_ddpm(z_T,return_z=True)
                    # text_list_ori = self.latent2text(z_k,return_list=True)
                    # z0_ori = z_k
                    # text_list_new = self.latent2text(1.25 * z_T,return_list=True)
                    # for tt in range(len(text_list_ori)):
                    #     input_list.append([sentent_list[tt].split()])
                    #     ori_list.append(text_list_ori[tt].split())
                    #     new_list.append(text_list_new[tt].split()) # [['it','is'],['a','v']]
                    # cos_sim_ = (z0_ori * z0_new).sum(1)/(z0_ori.norm(2,dim=1)*z0_new.norm(2,dim=1))
                    # cos_list.extend(cos_sim_.tolist())
                    # ori_norm_list.extend(z0_ori.norm(2,dim=1).tolist())
                    # new_norm_list.extend(z0_new.norm(2,dim=1).tolist())
            # from nltk.translate.bleu_score import corpus_bleu

            # ori_bleu=corpus_bleu(input_list,ori_list) * 100
            # new_bleu=corpus_bleu(input_list,new_list) * 100
            # print(ori_bleu,new_bleu)
            # print(new_bleu-ori_bleu)
            # import ipdb
            # ipdb.set_trace()
            
#             f.write('\n'.join([' '.join(i) for i in new_list]))
#######################
            # if True:
                    y1 = 1 - latent_labels.unsqueeze(1) # sentiment label
                    # y1 = torch.ones_like(latent_labels)
                    # y[:, 0] = y1
                    # y = y.unsqueeze(1)
                    # _, sample_time = self._sample_batch_repara_multiatt(self.sampler, y, z_k, f, repa, att_val)
                    if  interpolation:
                        _, sample_time = self._sample_batch_repara(self.sampler, y1.unsqueeze(1), z_k, f, repa)
                    else:
                        _, sample_time = self._sample_batch(self.sampler,y1,z_k,f)
                    
                    
                    # _, sample_time = self._sample_batch_repara_word(self.sampler, y, z_k, f, word=self.args.data_type)
                    # _, sample_time = self._sample_batch_repara_word(self.sampler, y, z_k, f, word='dog,dogs')
                    print(f'batch {j}, sampling time: {sample_time}')
            # f1.close()
            # f2.close()
            f.close()

    def get_samples_vec(self):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)

        # f1 = open(os.path.join(self.save_path, 'rec.txt'), 'w')
        # f2 = open(os.path.join(self.save_path, 'rec_ori.txt'), 'w')
        k = 3.5
        f = open(os.path.join(self.save_path, 'transfer_vec_op_repa' + str(k) + '.txt'), 'w')

        
        for j, batch in enumerate(self.test_data_batch):
            inputs, x1, tokenized_text_lengths = batch
            latent_labels = tokenized_text_lengths[:, -1]
            inputs = inputs.to(self.device)
            latent_labels1 = latent_labels.to(self.device)
            bert_fea = self.model_vae.encoder(inputs, attention_mask=(inputs > 0).float())[1]
            mu, logvar = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
            z_kk = mu.squeeze()
            num = z_kk.size(0)
            z_neg = z_kk[:num//2]
            z_pos = z_kk[num//2:]
            diff_vec = z_neg.mean(dim=0) - z_pos.mean(dim=0)
            neg2pos = z_neg - k * diff_vec
            pos2neg = z_pos + k * diff_vec
            all = torch.cat((neg2pos, pos2neg), 0)
            for jj in range(4):
                z_k = all[jj * 250:(jj + 1) * 250]
                latent_labels = latent_labels1[jj * 250:(jj + 1) * 250]
                ############################ begin
                sentent_list = []
                ########################### end
                for i in range(z_k.size(0)):
                    ori_text_x1 = \
                        self.decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                            '<EOS>')[0].replace('<BOS>', '').strip()
                    ori_text_x1 = ori_text_x1.split()
                    ori_text_x1 = ' '.join(ori_text_x1)
                    sentent_list.append(ori_text_x1)

                self.sentence_list = sentent_list
                ####################### end

                y = 1 - latent_labels
                y = y.unsqueeze(1)
                _, sample_time = self._sample_batch(None, y, z_k, f)


                # _, sample_time = self._sample_batch_repara(None, y, z_k, f)

                print(f'batch {j}, sampling time: {sample_time}')
        # f1.close()
        # f2.close()
        f.close()



class ConditionalSampling(ConditionalTransfer):
    def __init__(self, sampler, batch_size, latent_dim, n_classes, ccf, device, save_path, ode_kwargs,
                 ld_kwargs, sde_kwargs, every_n_plot=5, model_kwargs=None, gan=None, disable_bar=False, mode=None):
        self.n_classes = n_classes
        self.ccf = ccf
        self.save_path = save_path
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        self.plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

        self.sampler = partial(sampler, ccf=ccf, device=device, plot=self.plot, every_n_plot=every_n_plot,
                               **ode_kwargs, **ld_kwargs, **sde_kwargs, **model_kwargs)
        self.model_vae = model_kwargs['model']
        self.decoder_tokenizer = model_kwargs['dec_tokenizer']
        self.encoder_tokenizer = model_kwargs['enc_tokenizer']
        self.context_tokens = self.decoder_tokenizer.encode('<BOS>')

        self.args = model_kwargs['args']
        self.z_k = [torch.FloatTensor(batch_size, latent_dim).normal_(0, 1).to(device) for _ in range(1)]
        self.gan = gan
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = device
        self.sampling_num = model_kwargs['sampling_num']
        self.left = self.sampling_num % batch_size
        self.time = self.sampling_num // batch_size
        self.disable_bar = disable_bar
        self.wnl = WordNetLemmatizer()
        if self.left != 0:
            self.time += 1
        if mode[1] + mode[2] > 0:
            self.cls_tokenizer = AutoTokenizer.from_pretrained('../classifiers/sentiment')
            self.cls_model = AutoModelForSequenceClassification.from_pretrained(
                '../classifiers/sentiment').cuda().eval()

    def _sample_batch(self, sampler, y, save_path=None):
        start_sample_time = time.time()
        idx = str(y[0].item())
        f = open(os.path.join(self.save_path, 'sampling_' + idx + '.txt'), 'w')
        for z_k in self.z_k:
            if self.gan != None:
                z_k = self.gan.latent_generator(z_k)
            t1 = time.time()
            z = sampler(y=y, save_path=save_path, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z
            valid_lines = []
            t2 = time.time()
            print(t2 - t1)
            out = sample_sequence_conditional(
                model=self.model_vae.decoder,
                context=self.context_tokens,
                past=z.detach(),
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=z.size(0),
                device=self.device,
                decoder_tokenizer=self.decoder_tokenizer,
                eos_id=self.model_vae.eos_token_id
            )
            t3 = time.time()
            print(t3 - t2)
            for i in range(z.size(0)):
                text_x1 = \
                    self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                        '<EOS>')[0].replace('<BOS>', '').strip()
                text_x1 = text_x1.split()
                text_x1 = ' '.join(text_x1)
                valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
                f.write(text_x1 + '\n')
        f.close()
        t4 = time.time()
        print(t4 - t3)
        sample_time = time.time() - start_sample_time

        return valid_lines, sample_time

    def get_samples(self):
        # save_path = os.path.join(self.save_path, 'cond')
        os.makedirs(self.save_path, exist_ok=True)
        f = open(os.path.join(self.save_path, 'ori.txt'), 'w')
        for z_k in self.z_k:
            if self.gan != None:
                z_k = self.gan.latent_generator(z_k)
            valid_lines = []
            out = sample_sequence_conditional(
                model=self.model_vae.decoder,
                context=self.context_tokens,
                past=z_k.detach(),
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=z_k.size(0),
                device=self.device,
                decoder_tokenizer=self.decoder_tokenizer,
                eos_id=self.model_vae.eos_token_id
            )
            for i in range(z_k.size(0)):
                text_x1 = \
                    self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                        '<EOS>')[0].replace('<BOS>', '').strip()
                text_x1 = text_x1.split()
                text_x1 = ' '.join(text_x1)
                valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
                f.write(text_x1 + '\n')
        f.close()
        for i in range(self.n_classes):
            y = torch.tensor([i]).repeat(self.batch_size).to(self.device)

            valid_lines, sample_time = self._sample_batch(self.sampler, y, save_path=self.save_path)

            print(f'class {i}, sampling time: {sample_time}')

    def get_samples_length(self):
        # save_path = os.path.join(self.save_path, 'cond')
        os.makedirs(self.save_path, exist_ok=True)
        f = open(os.path.join(self.save_path, 'ori.txt'), 'w')
        for z_k in self.z_k:
            if self.gan != None:
                z_k = self.gan.latent_generator(z_k)
            valid_lines = []
            out = sample_sequence_conditional(
                model=self.model_vae.decoder,
                context=self.context_tokens,
                past=z_k.detach(),
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=z_k.size(0),
                device=self.device,
                decoder_tokenizer=self.decoder_tokenizer,
                eos_id=self.model_vae.eos_token_id
            )
            for i in range(z_k.size(0)):
                text_x1 = \
                    self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                        '<EOS>')[0].replace('<BOS>', '').strip()
                text_x1 = text_x1.split()
                text_x1 = ' '.join(text_x1)
                valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
                f.write(text_x1 + '\n')
        f.close()
        for i in [14]:
            y = torch.tensor([i / 14.]).repeat(self.batch_size).to(self.device)

            valid_lines, sample_time = self._sample_batch(self.sampler, y, save_path=self.save_path)
            print(f'class {i}, sampling time: {sample_time}')

    def target_generation(self, init_str, z_k, num=5, var=-1):
        mean = z_k.norm(dim=-1).mean()
        std = z_k.norm(dim=-1).std()
        z_init = self.text2latent(init_str)
        z_init = z_init / z_init.norm()
        logvar = var + torch.zeros_like(z_init)
        for i in range(num):
            norm_ = torch.FloatTensor(1).normal_(mean.item(), std.item()).cuda()
            z_final = z_init * norm_
            print(i)
            self.repara(z_final, logvar, 0, 5)

    def target_generation_1(self, init_str, z_k, var=-1):
        mean = z_k.norm(dim=-1).mean()
        std = z_k.norm(dim=-1).std()
        z_init = self.text2latent(init_str)
        # z_init = z_init / z_init.norm()  # 1 x latent_sz
        # logvar = var + torch.zeros_like(z_k) #
        # norm_  = torch.zeros_like(z_k[:,[0]]).normal_(mean.item(), std.item()).cuda() # 100 x 1
        # z_init = norm_.matmul(z_init)
        # z_final = self.model_vae.reparameterize(z_init, logvar, 1).squeeze(1)

        logvar = var + torch.zeros_like(z_init)  #1
        z_final = self.model_vae.reparameterize(z_init, logvar, z_k.shape[0]).squeeze()
        return z_final

    def _sample_batch_multiple(self, sampler, y, save_path=None, att_val=None):
        start_sample_time = time.time()
        idx = '_'.join(map(str, att_val))
        prefix = 'sampling_'
        f = open(os.path.join(self.save_path,
                              prefix + str(self.args.data_type)  + '_' + idx + '.txt'),
                 'w')
        sample_batch = self.batch_size
        pbar = trange(self.time, disable=self.disable_bar)
        for ii in pbar:
            if ii == self.time - 1 and self.left != 0:
                sample_batch = self.left
            if self.gan != None:
                z_k = self.gan.generate_z(sample_batch)

                def sequential_edit(text, label_y):
                    zz = self.text2latent(text)
                    label_y = torch.zeros_like(y[:1])  + label_y
                    z,ode_step = sampler(y=label_y,save_path=save_path, z_k=zz.clone())
                    logvar = -1 * torch.ones_like(z)
                    self.repara(z, logvar, 0, 20, ori_text=text)
                z, ode_step = sampler(y=y, save_path=save_path, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new
                pbar.set_description("Processing %s" % (str(ii) + ' ode_step: ' + str(ode_step)))
                valid_lines = []
                out = sample_sequence_conditional(
                    model=self.model_vae.decoder,
                    context=self.context_tokens,
                    past=z,
                    length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                    num_samples=z.size(0),
                    device=self.device,
                    decoder_tokenizer=self.decoder_tokenizer,
                    eos_id=self.model_vae.eos_token_id
                )

                for i in range(z.size(0)):
                    text_x1 = \
                        self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                            '<EOS>')[0].replace('<BOS>', '').strip()
                    text_x1 = text_x1.split()
                    text_x1 = ' '.join(text_x1)
                    valid_lines.append(text_x1)
                    f.write(text_x1 + '\n')

        f.close()
        sample_time = time.time() - start_sample_time
        print("sample time:",round(sample_time,2))
        return valid_lines, sample_time

    def _sample_batch_multiple_repa(self, sampler, y, save_path=None, att_val=None):
        start_sample_time = time.time()
        idx = '_'.join(map(str, att_val))
        f = open(os.path.join(self.save_path, 'sampling_' + str(self.args.weight_energy) + '_repa_' + idx + '.txt'),
                 'w')
        # for z_k in self.z_k:
        sample_batch = self.batch_size
        pbar = trange(self.time)
        for ii in pbar:
            if ii == self.time - 1 and self.left != 0:
                sample_batch = self.left
            # print('sampling', ii, 'th batch')
            # z_k = torch.FloatTensor(sample_batch, self.latent_dim).normal_(0, 1).to(self.device)
            if self.gan != None:
                # z_k = self.gan.latent_generator(z_k)
                z_k = self.gan.generate_z(sample_batch)
                z, ode_step = sampler(y=y, save_path=save_path, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new
                ######################
                with torch.no_grad():
                    logvar = -3 * torch.ones_like(z)
                    # logvar = logvar.clone()
                    new_z = self.model_vae.reparameterize(z, logvar, 10)  # bs x num x laten
                    new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
                    # sentent_list = [[]] * new_z.size(1)
                    bleu_list = [-200.] * new_z.size(0)
                    final_list = [''] * new_z.size(0)
                    final_tmp_list = [''] * new_z.size(0)
                    bleu_tmp_list = [-200] * new_z.size(0)
                    idx_list = [0.] * new_z.size(0)
                    # bleu_tmp = [-200.] * new_z.size(0)
                for ii in trange(new_z.size(1)):
                    z = new_z[:, ii]
                    out, neg_loss_list = sample_sequence_conditional_sample(
                        model=self.model_vae.decoder,
                        context=self.context_tokens,
                        past=z.detach(),
                        length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                        num_samples=z.size(0),
                        device=self.device,
                        decoder_tokenizer=self.decoder_tokenizer,
                        eos_id=self.model_vae.eos_token_id,
                        loss=True
                    )
                    bleu_tmp = neg_loss_list
                    text_tmp = []
                    flag_right = [False] * new_z.size(0)

                    # text_out = self.tokenizer(out,clean_up_tokenization_spaces=False, skip_special_tokens=True)
                    # for i in range(z.size(0)):
                    #     text_x1 = text_out[i]
                    #     text_x1_ = text_x1.strip()
                    #     text_tmp.append(text_x1)
                    #     for w in word.split(','):
                    #         if w in text_x1_:
                    #             flag_right[i] = True

                    for i in range(z.size(0)):
                        text_x1 = \
                            self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                                '<EOS>')[0].replace('<BOS>', '').strip()
                        text_x1_ = text_x1.split()
                        text_x1 = ' '.join(text_x1_)
                        text_tmp.append(text_x1)
                    # f.write('\n'.join(text_tmp))
                    # f.write('\n\n\n\n\n\n')
                    feature = self.cls_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                    logits = self.cls_model(**feature)[0]  # batch x 2
                    flag_right = logits.max(-1)[1] == y.squeeze()  # batch == y
                    for i in range(z.size(0)):
                        # if bleu_tmp[i] > bleu_list[i] and flag_right[i]:
                        #     bleu_list[i] = bleu_tmp[i]
                        #     final_list[i] = text_tmp[i]
                        #     idx_list[i] = ii
                        if flag_right[i]:
                            final_list[i] = text_tmp[i]
                            idx_list[i] = ii
                        if bleu_tmp[i] > bleu_tmp_list[i]:
                            bleu_tmp_list[i] = bleu_tmp[i]
                            final_tmp_list[i] = text_tmp[i]
                        # if ii == new_z.size(1) - 1:  # last one
                        #     if final_list[i] == '':
                        #         final_list[i] = final_tmp_list[i]
                    # sentent_list[ii].append(text_x1)
                for i in range(z.size(0)):
                    f.write(final_list[i] + '\n')
        f.close()
        sample_time = time.time() - start_sample_time

        return sample_time

    def _sample_batch_multiple_repa_mulwords(self, sampler, y, save_path=None, att_val=None):
        word = self.args.cls_.split(',')[-1]
        start_sample_time = time.time()
        idx = '_'.join(map(str, att_val))
        valid_lines = []
        repa_num = 20
        f = open(os.path.join(self.save_path,
                              'sampling_repa' + str(repa_num) + '_' + str(self.args.data_type) + '_' + str(
                                  self.args.cls_.replace(',','_')) + '_' + idx + '.txt'),'w')
        sample_batch = self.batch_size
        pbar = trange(self.time,disable=True)
        for ii in pbar:
            if ii == self.time - 1 and self.left != 0:
                sample_batch = self.left
            # print('sampling', ii, 'th batch')
            z_k = torch.FloatTensor(sample_batch, self.latent_dim).normal_(0, 1).to(self.device)
            if self.gan != None:
                z_k = self.gan.latent_generator(z_k)
                z, ode_step = sampler(y=y, save_path=save_path, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new
                ######################
                with torch.no_grad():
                    logvar = -3 * torch.ones_like(z)
                    new_z = self.model_vae.reparameterize(z, logvar, repa_num)  # bs x num x laten
                    new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
                    num_samples = new_z.size(0)
                    bleu_list = [-200.] * num_samples
                    final_list = [''] * num_samples
                    final_tmp_list = [''] * num_samples
                    bleu_tmp_list = [-200] * num_samples
                    idx_list = [0.] * num_samples
                    exist_times_list = [0] * num_samples
                    total_num_list = [0] * num_samples
                    exist_times_tmp = 0
                for ii in trange(new_z.size(1), disable=True):
                    z = new_z[:, ii]
                    out, neg_loss_list = sample_sequence_conditional(
                        model=self.model_vae.decoder,
                        context=self.context_tokens,
                        past=z.detach(),
                        length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                        num_samples=z.size(0),
                        device=self.device,
                        decoder_tokenizer=self.decoder_tokenizer,
                        eos_id=self.model_vae.eos_token_id,
                        loss=True
                    )
                    bleu_tmp = neg_loss_list
                    text_tmp = []
                    flag_right = [False] * new_z.size(0)
                    for i in range(z.size(0)):
                        text_x1 = \
                            self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                                '<EOS>')[0].replace('<BOS>', '').strip()
                        text_tmp.append(text_x1)
                        total_num, exist_times_tmp, flag_right[i] = self.multiple_token_exist_sentence(
                            target_words=word,
                            sentence=text_x1,
                            split_tag=',')
                    for i in range(z.size(0)):
                        if bleu_tmp[i] > bleu_list[i] and flag_right[i]:
                            bleu_list[i] = bleu_tmp[i]
                            final_list[i] = text_tmp[i]
                            idx_list[i] = ii
                        if exist_times_tmp > exist_times_list[i]:
                            if bleu_tmp[i] > bleu_tmp_list[i]:
                                bleu_tmp_list[i] = bleu_tmp[i]
                                final_tmp_list[i] = text_tmp[i]
                                exist_times_list[i] = exist_times_tmp
                        elif exist_times_tmp == exist_times_list[i] == 0: # 0
                            final_tmp_list[i] = text_tmp[i]
                        if bleu_tmp[i] > bleu_tmp_list[i]:
                            if exist_times_tmp > exist_times_list[i]:
                                bleu_tmp_list[i] = bleu_tmp[i]
                                final_tmp_list[i] = text_tmp[i]
                                exist_times_list[i] = exist_times_tmp

                        if ii == new_z.size(1) - 1:  # last one
                            if final_list[i] == '':
                                final_list[i] = final_tmp_list[i]
                for i in range(z.size(0)):
                    if final_list[i] == '':
                        continue
                    f.write(final_list[i] + '\n')
                    valid_lines.append(final_list[i])
        f.close()
        sample_time = time.time() - start_sample_time

        return sample_time,valid_lines

    def _multiple_repa_mulwords_attribut(self, sampler, y, save_path=None, att_val=None):
        word = self.args.cls_.split(',')[-1]
        start_sample_time = time.time()
        idx = '_'.join(map(str, att_val))
        repa_num = 20
        f = open(os.path.join(self.save_path,
                              'sampling_repav2' + str(repa_num) + '_' + str(self.args.data_type) + '_' + str(
                                  self.args.cls_.replace(',','_')) + '_' + idx + '.txt'),'w')
        sample_batch = self.batch_size
        pbar = trange(self.time,disable=True)
        for ii in pbar:
            if ii == self.time - 1 and self.left != 0:
                sample_batch = self.left
            # print('sampling', ii, 'th batch')
            z_k = torch.FloatTensor(sample_batch, self.latent_dim).normal_(0, 1).to(self.device)
            if self.gan != None:
                z_k = self.gan.latent_generator(z_k)
                z, ode_step = sampler(y=y, save_path=save_path, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new
                ######################
                with torch.no_grad():
                    logvar = -3 * torch.ones_like(z)
                    new_z = self.model_vae.reparameterize(z, logvar, repa_num)  # bs x num x laten
                    new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
                    num_samples = new_z.size(0)
                    bleu_list = [-200.] * num_samples
                    final_list = [''] * num_samples
                    final_tmp_list = [''] * num_samples
                    bleu_tmp_list = [-200] * num_samples
                    idx_list = [0.] * num_samples
                    exist_times_list = [0] * num_samples
                    total_num_list = [0] * num_samples
                    exist_times_tmp = 0
                for ii in trange(new_z.size(1), disable=True):
                    z = new_z[:, ii]
                    out, neg_loss_list = sample_sequence_conditional(
                        model=self.model_vae.decoder,
                        context=self.context_tokens,
                        past=z.detach(),
                        length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                        num_samples=z.size(0),
                        device=self.device,
                        decoder_tokenizer=self.decoder_tokenizer,
                        eos_id=self.model_vae.eos_token_id,
                        loss=True
                    )
                    bleu_tmp = neg_loss_list
                    text_tmp = []
                    flag_right = [False] * new_z.size(0)
                    for i in range(z.size(0)):
                        text_x1 = \
                            self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                                '<EOS>')[0].replace('<BOS>', '').strip()
                        text_tmp.append(text_x1)
                        total_num, exist_times_tmp, flag_right[i] = self.multiple_token_exist_sentence(
                            target_words=word,
                            sentence=text_x1,
                            split_tag=',')
                    ### attribute classifier
                    feature = self.cls_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                    logits = self.cls_model(**feature)[0]  # batch x 2
                    flag_right_attr = logits.max(-1)[1] == y[:,0].squeeze()  # batch == y


                    for i in range(z.size(0)):
                        if flag_right[i] and flag_right_attr[i]:
                            if bleu_tmp[i] > bleu_list[i]:
                                bleu_list[i] = bleu_tmp[i]
                                final_list[i] = text_tmp[i]
                                idx_list[i] = ii
                        elif flag_right_attr[i]:
                            if bleu_tmp[i] > bleu_tmp_list[i]:
                                bleu_tmp_list[i] = bleu_tmp[i]
                                final_tmp_list[i] = text_tmp[i]
                        elif flag_right[i]:
                            if bleu_tmp[i] > bleu_tmp_list[i]:
                                bleu_tmp_list[i] = bleu_tmp[i]
                                final_tmp_list[i] = text_tmp[i]
                        elif bleu_tmp[i] > bleu_tmp_list[i] and final_tmp_list[i] == '':
                            final_tmp_list[i] = text_tmp[i]
                        #
                        # if bleu_tmp[i] > bleu_tmp_list[i]:
                        #     if exist_times_tmp > exist_times_list[i]:
                        #         bleu_tmp_list[i] = bleu_tmp[i]
                        #         final_tmp_list[i] = text_tmp[i]
                        #         exist_times_list[i] = exist_times_tmp

                        if ii == new_z.size(1) - 1:  # last one
                            if final_list[i] == '':
                                final_list[i] = final_tmp_list[i]
                for i in range(z.size(0)):
                    if final_list[i] == '':
                        continue
                    f.write(final_list[i] + '\n')
        f.close()
        sample_time = time.time() - start_sample_time

        return sample_time

    def single_token_exist_sentence(self, target_word, sentence):
        tokens = word_tokenize(sentence)
        exist_times = 0
        for token in tokens:
            if self.wnl.lemmatize(token, 'n') == target_word or self.wnl.lemmatize(token, 'v') == target_word:
                exist_times += 1
        return exist_times  # target_word 一共出现在sentence的次数

    def multiple_token_exist_sentence(self, target_words, sentence, split_tag=','):
        words = target_words.split(split_tag)
        total_num = 0
        exist_times = 0
        for word in words:
            single_times = int(self.single_token_exist_sentence(word, sentence))
            total_num += single_times
            if single_times > 0:
                exist_times += 1
        success = True if exist_times == len(words) else False
        return total_num, exist_times, success  # total_num: 所有target words一共出现的次数; exist_times: 有几个target words出现在句子中

    # def multiple_token_exist_multiple_sentences(self, target_words, sentences):
    #     for sentence in sentences:
    #

    def get_samples_multiple(self, att_val_list='1,1,0',mode=[1,0,0], out_sentences=False):
        os.makedirs(self.save_path, exist_ok=True)
        for att_val in att_val_list:
            att_val = [float(i) for i in att_val.split(',')]
            y = torch.tensor([att_val] * self.batch_size).to(self.device)  # batch x num_att
            if mode[0] == 1:
                valid_lines, sample_time = self._sample_batch_multiple(self.sampler, y, save_path=self.save_path,
                                                                       att_val=att_val)
            if mode[1] == 1:
                sample_time = self._multiple_repa_mulwords_attribut(self.sampler, y, save_path=self.save_path,
                                                                       att_val=att_val)
            if mode[2] == 1:
                sample_time, valid_lines = self._sample_batch_multiple_repa_mulwords(self.sampler, y, save_path=self.save_path,
                                                                       att_val=att_val) # dog,dogs;ball,balls
        if out_sentences:
            return valid_lines
