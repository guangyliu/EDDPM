from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FFHQlmdb
from torch.utils.data import DataLoader


import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-T", '--time_steps', required=True, type=int)
parser.add_argument('-N', '--num_samples', type=int, default=-1)
parser.add_argument('--batch_size', default=420, type=int)
parser.add_argument("--device", default=0)

args = parser.parse_args()

with torch.no_grad():
    device = f'cuda:{args.device}'
    conf = ffhq128_autoenc_joint()
    conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
    conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
    conf.T_eval = 100
    conf.latent_T_eval = 100
    conf.name = 'eval'
    dataset = conf.make_dataset()
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # print(conf.name)
    model = LitModel(conf)
    # model.model = 
    # state = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/last.ckpt', map_location='cpu')
    model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt', map_location='cpu')
    # import ipdb; ipdb.set_trace()
    # print(model.load_state_dict(state['state_dict'], strict=False))
    # model.model = state
    model.ema_model = model.model
    model.to(device)
    model.eval()
    torch.manual_seed(4)
    
    use_inverted_noise = True
    
    print("Use Inverted Noise:", use_inverted_noise)

    sampler = model.conf._make_diffusion_conf(args.time_steps).make_sampler()
    
    scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        imgs = batch['img'].to(device)
        
        cond = model.encode(imgs)
        
        if use_inverted_noise:
            
            model_kwargs = {}
            
            if conf.model_type.has_autoenc():
                with torch.no_grad():
                    model_kwargs['cond'] = model.encode(imgs)
            
            x_T = sampler.ddim_reverse_sample_loop(
                model=model.model,
                x=imgs,
                clip_denoised=True,
                model_kwargs=model_kwargs)
            x_T = x_T['sample']
        
        else:
            x_T = torch.randn((len(imgs), 3, conf.img_size, conf.img_size),
                                  device=device)
        
        if conf.model_type == ModelType.ddpm:
            # the case where you want to calculate the inversion capability of the DDIM model
            assert use_inverted_noise
            pred_imgs = render_uncondition(
                conf=conf,
                model=model,
                x_T=x_T,
                sampler=sampler,
                latent_sampler=latent_sampler,
            )
        else:
            pred_imgs = render_condition(conf=conf,
                                            model=model.model,
                                            x_T=x_T,
                                            x_start=imgs,
                                            cond=cond,
                                            sampler=sampler)


        # (n, 1, 1, 1) => (n, )
        scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

        # need to normalize into [0, 1]
        norm_imgs = (imgs + 1) / 2
        norm_pred_imgs = (pred_imgs + 1) / 2
        # (n, )
        scores['ssim'].append(
            ssim(norm_imgs, norm_pred_imgs, size_average=False))
        # (n, )
        scores['mse'].append(
            (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
        # (n, )
        scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))

    for key in scores.keys():
        scores[key] = torch.cat(scores[key]).mean().float().item()
        
    print(scores)






















    #     original_img = batch['img']
    #     j = len(original_img) - 1
    #     if os.path.exists(f'./samples/reconstructions_best_fid_ema_T={args.time_steps}_N={args.num_samples}/images/image_{count+j}.png'): 
    #         count += len(original_img)
    #         continue
        
    #     cond = model.encode(original_img.to(device))
    #     # cond = model.model.encoder.forward(original_img)
        
        
    #     noise = torch.randn(len(original_img),
    #                         3,
    #                         conf.img_size,
    #                         conf.img_size,
    #                         device=device)
    #     pred_img = sampler.sample(model=model.model, noise=noise, cond=cond)
    #     pred_img = (pred_img + 1) / 2
    #     original_img = (original_img + 1) / 2

    #     for j, img in enumerate(pred_img):
    #         plt.imsave(f'./samples/reconstructions_best_fid_ema_T={args.time_steps}_N={args.num_samples}/images/image_{count+j}.png', img.cpu().permute([1, 2, 0]).numpy())
    #         if not os.path.exists(f'./samples/original_imgs/original_image_{count+j}.png'):
    #             plt.imsave(f'./samples/original_imgs/original_image_{count+j}.png', original_img[j].cpu().permute([1, 2, 0]).numpy())
        
    #     # import ipdb; ipdb.set_trace()
    #     count += len(pred_img)
    #     all_imgs.append(pred_img.cpu().numpy())

    #     if args.num_samples > 0 and count > args.num_samples: break

    # # all_imgs = np.concatenate(all_imgs, axis=0)
    # # np.save(f'./samples/reconstruction_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)