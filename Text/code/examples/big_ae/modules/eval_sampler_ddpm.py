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
from .eval_sampler import *
# from ..run_latent_generation import sample_sequence_conditional

class ConditionalTransferDDPM(ConditionalTransfer):
    def __init__(self, sampler, batch_size, latent_dim, n_classes, ccf, device, save_path, ode_kwargs,
                 ld_kwargs, sde_kwargs, every_n_plot=5, model_kwargs=None, test_data_batch=None, sampling=False, ddpm=None):
        # super().__init__(batch_size, latent_dim, n_classes, ccf, device, save_path)
        self.plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
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
        self.device = 'cuda' 
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
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained('../classifiers/sentiment')
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            '../classifiers/sentiment').cuda().eval()


    def _sample_batch(self, sampler, y, z_k, f):
        start_sample_time = time.time()
        if sampler:
            z = sampler(y=y, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z
        else:
            z = z_k.clone()
        valid_lines = []
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
        sentent_list = []
        for i in range(z.size(0)):
            text_x1 = \
                self.decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(
                    '<EOS>')[0].replace('<BOS>', '').strip()
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            valid_lines.append('[BOS] ' + text_x1 + ' [EOS]')
            if f:
                f.write(text_x1 + '\n')
            sentent_list.append(text_x1)
        sample_time = time.time() - start_sample_time
        if f:
            return valid_lines, sample_time
        else:
            return sentent_list,z
    def text2latent_ddpm(self, text):
        # tokenized_text0 = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(text))
        # tokenized_text0 = self.encoder_tokenizer.add_special_tokens_single_sentence(tokenized_text0)
        tokenized_text0 = self.encoder_tokenizer.encode(text)
        inputs_new = torch.tensor([tokenized_text0]).to(self.device)
        bert_fea = self.model_vae.encoder(inputs_new)[1]
        mu, _ = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
        mu = self.ddpm.add_noise(mu)
        return mu

    def latent2text_ddpm(self, new_z_k, loss=False, return_list = False, return_z = False):
        new_z_k = self.ddpm.sample_posterior(new_z_k, new_z_k.device, score_flag=2)
        loss_list = []
        text_list = []
        for i in range(new_z_k.size(0)):
            out,loss = sample_sequence_conditional(
                model=self.model_vae.decoder,
                context=self.context_tokens,
                past=new_z_k[i:i+1].detach(),
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=1,
                device=self.device,
                decoder_tokenizer=self.decoder_tokenizer,
                eos_id=self.model_vae.eos_token_id,
                loss= True
            )
            text_x1 = self.decoder_tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=False).split(
                '<EOS>')[0].replace('<BOS>', '').strip()
            text_x1 = text_x1.split()
            text_x1 = ' '.join(text_x1)
            out_loss = round(-np.mean(loss),3)
            loss_list.append(out_loss)
            text_list.append(text_x1)
            print(out_loss,text_x1)
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
    def interpolation_batch(self,z1,z2, num, return_text = False):
        theta = torch.arccos((z1 * z2).sum(1)/(torch.norm(z1,dim=1)*torch.norm(z2,dim=1)))
        sin_theta = torch.sin(theta)
    
        loss_list_all = []
        text = ''
        text_list_all = []
        for i in range(num):
            j = 1.* i / num
            tmp1 = torch.matmul(torch.diag(torch.sin((1-j)*theta)/sin_theta) , z1)
            tmp2 = torch.matmul(torch.diag(torch.sin(j*theta)/sin_theta) ,z2)
            z_tmp = tmp1 + tmp2 #torch.sin((1-j)*theta)/sin_theta * z1 + torch.sin(j*theta)/sin_theta *z2
            print(i)
            text_list, loss_list = self.latent2text_ddpm(z_tmp,loss=True,return_list=True)
            text_list_all.append(text_list)  # num * batch
            loss_list_all.append(loss_list)
            # text+= (str(i)+'\t'+str(round(loss,2))+'\t'+a+'\n')
        
        for j in range(z_tmp.size(0)):
            for i in range(num):
                text+= (str(i)+'\t'+str(round(loss_list_all[i][j],2)).ljust(4,' ')+'\t'+text_list_all[i][j]+'\n')
            text+='\n'
        if return_text:
            return loss_list_all, text
        return loss_list_all
    def linear_interpolation_batch(self,z1,z2, num, return_text=False):

        loss_list_all = []
        text = ''
        text_list_all = []
        z_diff = z2 - z1
        for i in range(num):
            j = 1.* i / num
            z_tmp = z1 + j * z_diff
            print(i)
            text_list, loss_list = self.latent2text_ddpm(z_tmp,loss=True, return_list=True)
            text_list_all.append(text_list)  # num * batch
            loss_list_all.append(loss_list)
        for j in range(z_tmp.size(0)):
            for i in range(num):
                text+= (str(i)+'\t'+str(round(loss_list_all[i][j],2)).ljust(4,' ')+'\t'+text_list_all[i][j]+'\n')
            text+='\n'
        if return_text:
            return loss_list_all, text
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

    def _sample_batch_repara(self, sampler, y, z_k, f, repa_num=3):
        start_sample_time = time.time()
        z_k = self.ddpm.add_noise(z_k)
        # self.interpolation(z_k[0],z_k[2],15)
        
        ######################
        ori_loss_list = []
        batch = z_k.shape[0]
        print('batch',batch)
        all_loss_list = []
        all_linear_list = []
        slerp_text = ''
        lerp_text = ''
        num =15
        int_arr =  []
        from tqdm import trange
        half_num = z_k.shape[0] // 2
        # for i in trange(batch-1):
            
        loss_list,slerp_ = self.interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True)
        linear_list, lerp_ = self.linear_interpolation_batch(z_k[:half_num],z_k[-half_num:],num,True)
        if len(int_arr) == 0:
            int_arr = np.array(loss_list).T
            int_arr_linear = np.array(linear_list).T
        else:
            int_arr=np.append(int_arr,np.array(loss_list).T,0)
            linear_list = np.append(int_arr_linear,np.array(linear_list).T,0)
        # all_loss_list.extend(loss_list)
        # all_linear_list.extend(linear_list)
        slerp_text += (slerp_+'\n')
        lerp_text += (lerp_+'\n')
        # int_arr = np.array(all_loss_list)

        out_arr = np.array([int_arr.mean(0),int_arr.var(0)])

        # int_arr_linear = np.array(all_linear_list)
        out_linear_arr = np.array([int_arr_linear.mean(0),int_arr_linear.var(0)])
        # with open('int_arr_ll.npy', 'wb') as f:
        #     np.save(f, int_arr)
        with open('out_arr_slerp.txt', 'w') as f:
            for i in range(out_arr.shape[1]):
                f.write(str(out_arr[0,i])+'\t'+str(out_arr[1,i])+'\n')
            f.write(slerp_text)
        with open('out_arr_lerp.txt', 'w') as f:
            for i in range(out_linear_arr.shape[1]):
                f.write(str(out_linear_arr[0,i])+'\t'+str(out_linear_arr[1,i])+'\n')
            f.write(lerp_text)

############
        # ori_loss_list = []
        # batch = z_k.shape[0]
        # print('batch',batch)
        # all_loss_list = []
        # all_linear_list = []
        # slerp_text = ''
        # lerp_text = ''
        # z_k = torch.randn_like(z_k)
        # num =15
        # batch=z_k.size(0)
        # int_arr = []
        # from tqdm import trange
        # for i in trange(batch-1):
        #     # for j in range(i+1,batch):
        #     loss_list,slerp_ = self.interpolation_batch(z_k[i].repeat(batch-i-1,1),z_k[i+1:],num,True)
        #     linear_list, lerp_ = self.linear_interpolation_batch(z_k[i].repeat(batch-i-1,1),z_k[i+1:],num,True)
        #     if len(int_arr) == 0:
        #         int_arr = np.array(loss_list).T
        #         int_arr_linear = np.array(linear_list).T
        #     else:
        #         int_arr=np.append(int_arr,np.array(loss_list).T,0)
        #         linear_list = np.append(int_arr_linear,np.array(linear_list).T,0)
        #     # all_loss_list.extend(loss_list)
        #     # all_linear_list.extend(linear_list)
        #     slerp_text += slerp_
        #     lerp_text += lerp_
        # # int_arr = np.array(all_loss_list)

        # out_arr = np.array([int_arr.mean(0),int_arr.var(0)])

        # # int_arr_linear = np.array(all_linear_list)
        # out_linear_arr = np.array([int_arr_linear.mean(0),int_arr_linear.var(0)])
        # # with open('int_arr_ll.npy', 'wb') as f:
        # #     np.save(f, int_arr)
        # with open('out_arr_slerp_sample.txt', 'w') as f:
        #     for i in range(out_arr.shape[1]):
        #         f.write(str(out_arr[0,i])+'\t'+str(out_arr[1,i])+'\n')
        #     f.write(slerp_text)
        # with open('out_arr_lerp_sample.txt', 'w') as f:
        #     for i in range(out_linear_arr.shape[1]):
        #         f.write(str(out_linear_arr[0,i])+'\t'+str(out_linear_arr[1,i])+'\n')
        #     f.write(lerp_text)

        import sys
        sys.exit()

        #######################
        if sampler:
            z = sampler(y=y, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z: bs x latent
        else:
            z = z_k.clone()
        
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
            new_z = self.ddpm.sample_posterior(new_z.view(-1,self.args.latent_size), new_z.device).view(z.size(0),-1,self.args.latent_size)
            # sentent_list = [[]] * new_z.size(1)
            bleu_list = [0.] * new_z.size(0)
            final_list = [''] * new_z.size(0)
            final_tmp_list = [''] * new_z.size(0)
            bleu_tmp_list = [-0.1] * new_z.size(0)
            idx_list = [0.] * new_z.size(0)

            for ii in trange(new_z.size(1)):
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
    def _sample_batch_repara_multiatt(self, sampler, y, z_k, f, repa_num=5, att_val=None):
        # att_val = [float(i) for i in att_val.split(',')]
        start_sample_time = time.time()
        z_k = self.ddpm.add_noise(z_k)
        z = sampler(y=y, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z: bs x latent
        with torch.no_grad():
            logvar = -3 * torch.ones_like(z)
            # logvar = logvar.clone()
            new_z = self.model_vae.reparameterize(z, logvar, repa_num)  # bs x num x latent
            ########################### end
            new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
            new_z = self.ddpm.sample_posterior(new_z.view(-1,self.args.latent_size), new_z.device).view(z.size(0),-1,self.args.latent_size)
            # sentent_list = [[]] * new_z.size(1)
            bleu_list = [0.] * new_z.size(0)
            final_list = [''] * new_z.size(0)
            final_tmp_list = [''] * new_z.size(0)
            bleu_tmp_list = [-0.1] * new_z.size(0)
            idx_list = [0.] * new_z.size(0)
            all_flag = [-1] * new_z.size(0)
            for ii in trange(new_z.size(1)):
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
                flag_right = logits.max(-1)[1] == y[:,0]  # batch == y
                t4 = time.time()
                if len(self.args.cls_step.split(','))==1:
                    t_feature = self.tense_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                    t_logits = self.tense_model(**t_feature)[0]
                    t_flag_right = t_logits.max(-1)[1] == 0 #y[:, 1]
                    for i in range(z.size(0)):
                        if bleu_tmp[i] > bleu_list[i] and flag_right[i] and t_flag_right[i]:
                            bleu_list[i] = bleu_tmp[i]
                            final_list[i] = text_tmp[i]
                            idx_list[i] = ii
                        if bleu_tmp[i] > bleu_tmp_list[i] and (flag_right[i] or t_flag_right[i]):
                            bleu_tmp_list[i] = bleu_tmp[i]
                            final_tmp_list[i] = text_tmp[i]
                        elif bleu_tmp[i] > bleu_tmp_list[i] and (bleu_tmp_list[i]<0):
                            bleu_tmp_list[i] = bleu_tmp[i]
                            final_tmp_list[i] = text_tmp[i]
                        if ii == new_z.size(1) - 1:  # last one
                            if final_list[i] == '':
                                final_list[i] = final_tmp_list[i]
                elif len(self.args.cls_step.split(',') )==3:
                    t_feature = self.tense_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                    t_logits = self.tense_model(**t_feature)[0]
                    t_flag_right = t_logits.max(-1)[1] == y[:, 1]
                    f_feature = self.formal_tokenizer(text_tmp, padding=True, return_tensors="pt").to(y.device)
                    f_logits = self.formal_model(**f_feature)[0]
                    f_flag_right = f_logits.max(-1)[1] == y[:, 1]
                    all_flag_tmp= flag_right[i] + t_flag_right[i] + f_flag_right[i]
                    for i in range(z.size(0)):
                        if all_flag_tmp > all_flag[i]:
                            all_flag[i] = all_flag_tmp
                            if bleu_tmp[i] > bleu_list[i]:
                                bleu_list[i] = bleu_tmp[i]
                                final_list[i] = text_tmp[i]
                                idx_list[i] = ii
                            elif bleu_tmp[i] > bleu_tmp_list[i] :
                                bleu_tmp_list[i] = bleu_tmp[i]
                                final_tmp_list[i] = text_tmp[i]
                        # elif bleu_tmp[i] > bleu_tmp_list[i] and (bleu_tmp_list[i]<0):
                        #     bleu_tmp_list[i] = bleu_tmp[i]
                        #     final_tmp_list[i] = text_tmp[i]
                        if ii == new_z.size(1) - 1:  # last one
                            if final_list[i] == '':
                                final_list[i] = final_tmp_list[i]

            for i in range(z.size(0)):
                f.write(final_list[i] + '\n')
        sample_time = time.time() - start_sample_time
        return None, sample_time

    def _sample_batch_repara_word(self, sampler, y, z_k, f, word=''):
        start_sample_time = time.time()
        y = torch.ones_like(y)
        if sampler:
            z = sampler(y=y, z_k=z_k.clone())  # 进入 sample_q_ode 函数, 得到new z: bs x latent
        else:
            z = z_k.clone()
        with torch.no_grad():
            logvar = -3 * torch.ones_like(z)
            # logvar = logvar.clone()
            new_z = self.model_vae.reparameterize(z, logvar, 5)  # bs x num x latent
            ########################### end
            new_z = torch.cat((z.clone().unsqueeze(1), new_z), 1)
            # sentent_list = [[]] * new_z.size(1)
            bleu_list = [-200.] * new_z.size(0)
            final_list = [''] * new_z.size(0)
            final_tmp_list = [''] * new_z.size(0)
            bleu_tmp_list = [-200.] * new_z.size(0)
            idx_list = [0.] * new_z.size(0)
            for ii in trange(new_z.size(1)):
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
                    text_x1_ = text_x1.split()
                    text_x1 = ' '.join(text_x1_)
                    text_tmp.append(text_x1)


                    for w in word.split(','):
                        if w in text_x1_:
                            flag_right[i] = True

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
                            final_list[i] = final_tmp_list[i]
                # sentent_list[ii].append(text_x1)
            for i in range(z.size(0)):
                f.write(final_list[i] + '\n')
            sample_time = time.time() - start_sample_time
        return None, sample_time

    def get_samples(self, desired_label=0.0):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)

        # f1 = open(os.path.join(self.save_path, 'rec.txt'), 'w')
        # f2 = open(os.path.join(self.save_path, 'rec_ori.txt'), 'w')
        # f = open(os.path.join(self.save_path, 'transfer_'+self.args.data_type+'_test_' + str(self.args.reg_logits) + '.txt'), 'w')
        repa = self.args.repa_num
        sym = ''
        if repa > 0:
            sym = 'repa' + str(repa) + '_'
        if self.sampling:
            file_name = os.path.join(self.save_path,
                                     'sampling_' + '_' + str(desired_label) + '.0.txt')
        else:
            file_name = os.path.join(self.save_path,
                                     'transfer_' + sym  + '.txt')
        f = open(file_name, 'w')
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

                y = 1 - latent_labels
                # y = torch.ones_like(latent_labels)
                y = y.unsqueeze(1)
                if repa == 0:
                    _, sample_time = self._sample_batch(self.sampler, y, z_k, f)  #
                elif repa > 0:
                    _, sample_time = self._sample_batch_repara(self.sampler, y, z_k, f, repa)
                # _, sample_time = self._sample_batch_repara_word(self.sampler, y, z_k, f, word=self.args.data_type)
                # _, sample_time = self._sample_batch_repara_word(self.sampler, y, z_k, f, word='dog,dogs')
                print(f'batch {j}, sampling time: {sample_time}')
        # f1.close()
        # f2.close()
        f.close()
    def get_samples_multiple(self,att_val_list='1,1,1'):
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
#                     z_T = self.ddpm.add_noise(z_k)
#                     import ipdb
#                     ipdb.set_trace()
#                     # text_list_ori, z0_ori = self.latent2text_ddpm(z_T,return_z=True)
#                     text_list_ori = self.latent2text(z_k,return_list=True)
#                     z0_ori = z_k
#                     text_list_new, z0_new = self.latent2text_ddpm(1.3 * z_T,return_z=True)
#                     for tt in range(len(text_list_ori)):
#                         input_list.append([sentent_list[tt].split()])
#                         ori_list.append(text_list_ori[tt].split())
#                         new_list.append(text_list_new[tt].split()) # [['it','is'],['a','v']]
#                     cos_sim_ = (z0_ori * z0_new).sum(1)/(z0_ori.norm(2,dim=1)*z0_new.norm(2,dim=1))
#                     cos_list.extend(cos_sim_.tolist())
#                     ori_norm_list.extend(z0_ori.norm(2,dim=1).tolist())
#                     new_norm_list.extend(z0_new.norm(2,dim=1).tolist())
#             from nltk.translate.bleu_score import corpus_bleu

#             ori_bleu=corpus_bleu(input_list,ori_list) * 100
#             new_bleu=corpus_bleu(input_list,new_list) * 100
#             print(ori_bleu,new_bleu)
#             print(new_bleu-ori_bleu)

            
#             f.write('\n'.join([' '.join(i) for i in new_list]))
#######################
                    if True:
                        y1 = 1 - latent_labels # sentiment label
                        # y1 = torch.ones_like(latent_labels)
                        # y[:, 0] = y1
                        # y = y.unsqueeze(1)
                        # _, sample_time = self._sample_batch_repara_multiatt(self.sampler, y, z_k, f, repa, att_val)
                        _, sample_time = self._sample_batch_repara(self.sampler, y1.unsqueeze(1), z_k, f, repa)
                        # _, sample_time = self._sample_batch(self.sampler,y,z_k,f)
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
        f = open(os.path.join(self.save_path, 'transfer_vec_op_repa' + str(self.args.reg_logits) + '.txt'), 'w')

        k = 2.5
        for j, batch in enumerate(self.test_data_batch):
            inputs, x1, tokenized_text_lengths = batch
            latent_labels = tokenized_text_lengths[:, -1]
            inputs = inputs.to(self.device)
            latent_labels1 = latent_labels.to(self.device)
            bert_fea = self.model_vae.encoder(inputs, attention_mask=(inputs > 0).float())[1]
            mu, logvar = self.model_vae.encoder.linear(bert_fea).chunk(2, -1)
            import ipdb
            z_kk = mu.squeeze()
            z_neg = z_kk[:500]
            z_pos = z_kk[500:]
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
                # _, sample_time = self._sample_batch(None, y, z_k, f)


                _, sample_time = self._sample_batch_repara(None, y, z_k, f)

                print(f'batch {j}, sampling time: {sample_time}')
        # f1.close()
        # f2.close()
        f.close()

