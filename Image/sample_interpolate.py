from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-T", '--time_steps', required=True, type=int)
parser.add_argument('-N', '--num_samples', required=True, type=int)
parser.add_argument('--dataset', type=str, default='ffhq')
parser.add_argument('--batch_size', default=296, type=int)
parser.add_argument("--device", default=0)
parser.add_argument('--ckpt', default='best_ema')

args = parser.parse_args()

with torch.no_grad():
    device = f'cuda:{args.device}'
    if args.dataset == 'ffhq':
        conf = ffhq128_autoenc_joint()
    elif args.dataset == 'horse':
        conf = horse128_autoenc_joint()
        conf.pretrain = PretrainConfig(
            name='200M',
            path=f'ckpts/horse128_autoenc/last.ckpt', # ae + ldm
            # path=f'checkpoints/ffhq128_autoenc/last_aeonly.ckpt', # only ae
            # path=f'checkpoints/ciai_bk/last.ckpt'
            # path=f'checkpoints/diffae_ckpt/v3/celeba64d2c_2gpu_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt'
        )
        conf.latent_infer_path = None
    conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
    conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
    conf.name = 'eval'
    # print(conf.name)
    model = LitModel(conf)
    if args.dataset == 'ffhq':
        model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt', map_location='cpu')
    elif args.dataset == 'horse':
        if args.ckpt == 'best_ema':
            model.model = torch.load("./checkpoints/diffae_ckpt/v3/horse_4gpu_b100_l0_lc_w1.8_scale_ft_teval100/fid_ckpt/best_fid_ema_model.ckpt", map_location='cpu')
        elif args.ckpt == 'best':
            model.model = torch.load("./checkpoints/diffae_ckpt/v3/horse_4gpu_b100_l0_lc_w1.8_scale_ft_teval100/fid_ckpt/best_fid_model.ckpt", map_location='cpu')
        
    model.ema_model = model.model
    model.to(device)
    torch.manual_seed(42)

    os.makedirs(f"./samples/{args.dataset}", exist_ok=True)

    folder_str = 'samples_interpolation_best_fid' if args.ckpt == 'best_ema' else "sample_best_single_fid"

    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}', exist_ok=True)
    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/images', exist_ok=True)
    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/numpys', exist_ok=True)

    count = 0
    all_imgs = []
    for i in tqdm(range(args.num_samples // args.batch_size + 1), total=args.num_samples // args.batch_size + 1):
        if count == args.num_samples:
            break
            
        count += args.batch_size
        if count > args.num_samples:
            count -= args.batch_size
            nums = args.num_samples - count
            count += nums
        else:
            nums = args.batch_size
        
        # j = nums - 1
        # if os.path.exists(f'./samples/samples_interpolation_best_fid_T={args.time_steps}_N={args.num_samples}/images/image_{count-nums+j}.png'): 
        #     continue
        # if i < 85: continue
        
        if nums < 0:
            import ipdb; ipdb.set_trace()
        
        x_T = torch.randn(nums,
                            3,
                            conf.img_size,
                            conf.img_size,
                            device=device)
        
        
        if conf.train_mode == TrainMode.latent_diffusion or conf.train_mode == TrainMode.joint: 
            latent_noise_1 = torch.randn(len(x_T), conf.style_ch, device=device)
            latent_noise_2 = torch.randn(len(x_T), conf.style_ch, device=device)
        else:
            raise NotImplementedError()

        sampler = conf._make_diffusion_conf(args.time_steps).make_sampler()
        latent_sampler = conf._make_latent_diffusion_conf(args.time_steps).make_sampler()

        cond_1 = latent_sampler.sample(
            model=model.model.latent_net,
            noise=latent_noise_1,
            clip_denoised=conf.latent_clip_sample,
        )
        
        cond_2 = latent_sampler.sample(
            model=model.model.latent_net,
            noise=latent_noise_2,
            clip_denoised=conf.latent_clip_sample,
        )
        
        if conf.latent_znormalize:
            cond_1 = cond_1 * model.conds_std.to(device) + model.conds_mean.to(device)
            cond_2 = cond_2 * model.conds_std.to(device) + model.conds_mean.to(device)
            
        
        for mix_ratio in [0.0, 0.1]:
            cond = cond_1 * mix_ratio + (1 - mix_ratio) * cond_2
            pred_img = sampler.sample(model=model.model, noise=x_T, cond=cond)
            pred_img = (pred_img + 1) / 2
        
            for j, img in enumerate(pred_img):
                os.makedirs(f'./samples/samples_interpolation_best_fid_T={args.time_steps}_N={args.num_samples}/batch_{count-nums+j}', exist_ok=True)
                plt.imsave(f'./samples/samples_interpolation_best_fid_T={args.time_steps}_N={args.num_samples}/batch_{count-nums+j}/interpolation_{mix_ratio}.png', img.cpu().permute([1, 2, 0]).numpy())
            
        # count += len(pred_img)
        


        


        # imgs = model.sample(nums, device=device, T=args.time_steps, T_latent=args.time_steps)
        # all_imgs.append(imgs.cpu().numpy())
        
        # for j in range(nums):
        #     plt.imsave(f'./samples/samples_interpolation_best_fid_T={args.time_steps}_N={args.num_samples}/images/image_{count-nums+j}.png', imgs[j].cpu().permute([1, 2, 0]).numpy())

    # all_imgs = np.concatenate(all_imgs, axis=0)
    # np.save(f'./samples/samples_interpolation_best_fid_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)