from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from dataset import FFHQlmdb
from torch.utils.data import DataLoader, Subset


parser = argparse.ArgumentParser()
parser.add_argument("-T", '--time_steps', required=True, type=int)
parser.add_argument('-N', '--num_samples', type=int, default=-1)
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
    conf.T_eval = 100
    conf.latent_T_eval = 100
    conf.name = 'eval'
    dataset = conf.make_dataset()
    
    num_samples = len(dataset)
    indices = list(range(num_samples))
    # np.random.shuffle(indices)
    
    # First 500 random indices
    first_500_indices = indices[:args.num_samples]

    # Remaining indices after the first 500
    second_500_indices = indices[1:args.num_samples+1]

    # Subset of the first 500 random images
    subset1 = Subset(dataset, first_500_indices)

    # Subset of the second 500 random images (non-overlapping)
    subset2 = Subset(dataset, second_500_indices)

    # Create data loaders for the subsets
    dataloader_subset1 = DataLoader(subset1, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dataloader_subset2 = DataLoader(subset2, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # print(conf.name)
    model = LitModel(conf)
    if args.dataset == 'ffhq':
        model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt', map_location='cpu')
    elif args.dataset == 'horse':
        if args.ckpt == 'best_ema':
            model.model = torch.load("./checkpoints/diffae_ckpt/v3/horse_4gpu_b100_l0_lc_w1.8_scale_ft_teval100/fid_ckpt/best_fid_ema_model.ckpt", map_location='cpu')
        elif args.ckpt == 'best':
            model.model = torch.load("./checkpoints/diffae_ckpt/v3/horse_4gpu_b100_l0_lc_w1.8_scale_ft_teval100/fid_ckpt/best_fid_model.ckpt", map_location='cpu')
        elif args.ckpt == 'bs':
            pass
    
    model.ema_model = model.model
    model.to(device)
    model.eval()
    torch.manual_seed(42)

    os.makedirs(f"./samples/{args.dataset}", exist_ok=True)

    if args.ckpt == 'best_ema':
        folder_str = 'interpolation_best_ema_fid'
    elif args.ckpt == 'best':
        folder_str = "interpolation_best_fid"
    else:
        folder_str = "interpolation_bs"
    # folder_str = 'interpolation_best_ema_fid' if args.ckpt == 'best_ema' else "interpolation_best_fid"

    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}', exist_ok=True)
    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/images', exist_ok=True)
    os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/numpys', exist_ok=True)

    print("start running")

    count = 0
    all_imgs = []
    for batch1, batch2 in tqdm(zip(dataloader_subset1, dataloader_subset2), total=len(dataloader_subset1)):
        original_img_1 = batch1['img']
        original_img_2 = batch2['img']


        if os.path.exists(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/batch_{count+len(original_img_1) - 1}/interpolation_0.4.png'):
            count += len(original_img_1)
            continue


        cond_1 = model.encode(original_img_1.to(device))
        cond_2 = model.encode(original_img_2.to(device))

        sampler = model.conf._make_diffusion_conf(args.time_steps).make_sampler()
        
        noise = torch.randn(len(original_img_1),
                            3,
                            conf.img_size,
                            conf.img_size,
                            device=device)
        
        original_img_1 = (original_img_1 + 1) / 2
        original_img_2 = (original_img_2 + 1) / 2


        for mix_ratio in [0.2, 0.4, 0.6, 0.8]:
        # for mix_ratio in [0.0]:
            cond = cond_1 * mix_ratio + (1 - mix_ratio) * cond_2
            # cond = cond * 0.9
            pred_img = sampler.sample(model=model.model, noise=noise, cond=cond)
            pred_img = (pred_img + 1) / 2
        
            for j, img in enumerate(pred_img):
                os.makedirs(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/batch_{count+j}', exist_ok=True)
                plt.imsave(f'./samples/{args.dataset}/{folder_str}_T={args.time_steps}_N={args.num_samples}/batch_{count+j}/interpolation_{mix_ratio}.png', img.cpu().permute([1, 2, 0]).numpy())
            
        count += len(pred_img)

    # all_imgs = np.concatenate(all_imgs, axis=0)
    # np.save(f'./samples/interpolation_best_fid_ckpt_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)