from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-T", '--time_steps', required=True, type=int)
parser.add_argument('-N', '--num_samples', required=True, type=int)
parser.add_argument('--batch_size', default=296, type=int)
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
    conf.latent_znormalize = False
    # print(conf.name)
    model = LitModel(conf)
    model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt', map_location='cpu')
    model.ema_model = model.model
    # state = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/last.ckpt', map_location='cpu')
    # import ipdb; ipdb.set_trace()
    # print(model.load_state_dict(state['state_dict'], strict=False))
    # import ipdb; ipdb.set_trace()
    # print(model.load_state_dict(state['state_dict'], strict=False))
    # model.model = state
    model.to(device)
    torch.manual_seed(4)

    os.makedirs(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}', exist_ok=True)
    os.makedirs(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}/images', exist_ok=True)
    os.makedirs(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}/numpys', exist_ok=True)

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
        
        j = nums - 1
        # if os.path.exists(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}/images/image_{count-nums+j}.png'): 
        #     continue
        
        imgs = model.sample(nums, device=device, T=args.time_steps, T_latent=args.time_steps)
        all_imgs.append(imgs.cpu().numpy())
        
        for j in range(nums):
            plt.imsave(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}/images/image_{count-nums+j}.png', imgs[j].cpu().permute([1, 2, 0]).numpy())

    # all_imgs = np.concatenate(all_imgs, axis=0)
    # np.save(f'./samples/samples_no_norm_best_fid_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)