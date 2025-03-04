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
parser.add_argument('--batch_size', default=420, type=int)
parser.add_argument("--device", default=0)

args = parser.parse_args()

with torch.no_grad():
    device = f'cuda:{args.device}'
    conf = celeba64d2c_ddpm()
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
    # model.model = 
    state = torch.load(f"checkpoints/celeba64d2c_ddpm/last.ckpt", map_location='cpu')
    # state = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/last.ckpt', map_location='cpu')
    # model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_ema_model.ckpt', map_location='cpu')
    # import ipdb; ipdb.set_trace()
    print(model.load_state_dict(state['state_dict'], strict=False))
    
    # model.model = state
    model.ema_model = model.model
    model.to(device)
    model.eval()
    torch.manual_seed(4)

    os.makedirs(f'./samples/celeba_interpolation_ddpm_T={args.time_steps}_N={args.num_samples}', exist_ok=True)

    count = 0
    all_imgs = []
    sampler = model.conf._make_diffusion_conf(args.time_steps).make_sampler()
    for batch1, batch2 in tqdm(zip(dataloader_subset1, dataloader_subset2), total=len(dataloader_subset1)):
        original_img_1 = batch1['img']
        original_img_2 = batch2['img']
        
        x_T_1 = sampler.ddim_reverse_sample_loop(
                model=model.model,
                x=original_img_1.to(device),
                clip_denoised=True)
        x_T_1 = x_T_1['sample']
        
        x_T_2 = sampler.ddim_reverse_sample_loop(
                model=model.model,
                x=original_img_2.to(device),
                clip_denoised=True)
        x_T_2 = x_T_2['sample']

        original_img_1 = (original_img_1 + 1) / 2
        original_img_2 = (original_img_2 + 1) / 2

        for mix_ratio in [0.2, 0.4]:
            x_T = x_T_1 * mix_ratio + (1 - mix_ratio) * x_T_2
            pred_img = sampler.sample(model=model.model, noise=x_T)
            pred_img = (pred_img + 1) / 2
        
            for j, img in enumerate(pred_img):
                os.makedirs(f'./samples/celeba_interpolation_ddpm_T={args.time_steps}_N={args.num_samples}/batch_{count+j}', exist_ok=True)
                plt.imsave(f'./samples/celeba_interpolation_ddpm_T={args.time_steps}_N={args.num_samples}/batch_{count+j}/interpolation_{mix_ratio}_0.9.png', img.cpu().permute([1, 2, 0]).numpy())
            
        count += len(pred_img)

    # all_imgs = np.concatenate(all_imgs, axis=0)
    # np.save(f'./samples/celeba_interpolation_ddpm_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)