from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FFHQlmdb
from torch.utils.data import DataLoader


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-T", '--time_steps', required=True, type=int)
parser.add_argument('-N', '--num_samples', type=int, default=-1)
parser.add_argument('--batch_size', default=296, type=int)
parser.add_argument("--device", default=0)

args = parser.parse_args()

with torch.no_grad():
    device = f'cuda:{args.device}'
    conf = celeba64d2c_autoenc_joint()
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
    # import ipdb; ipdb.set_trace()
    # print(model.load_state_dict(state['state_dict'], strict=False))
    # model.model = state
    model.to(device)
    torch.manual_seed(4)

    os.makedirs(f'./samples/celeba_reconstructions_bs_T={args.time_steps}_N={args.num_samples}', exist_ok=True)
    os.makedirs(f'./samples/celeba_reconstructions_bs_T={args.time_steps}_N={args.num_samples}/images', exist_ok=True)

    count = 0
    # all_imgs = []
    for batch in tqdm(dataloader):

        original_img = batch['img']
        j = len(original_img) - 1
        if os.path.exists(f'./samples/celeba_reconstructions_bs_T={args.time_steps}_N={args.num_samples}/images/image_{count+j}.png'): 
            count += len(original_img)
            continue
        
        cond = model.encode(original_img.to(device))
        sampler = model.conf._make_diffusion_conf(args.time_steps).make_sampler()
        noise = torch.randn(len(original_img),
                            3,
                            conf.img_size,
                            conf.img_size,
                            device=device)
        pred_img = sampler.sample(model=model.model, noise=noise, cond=cond)
        pred_img = (pred_img + 1) / 2
        original_img = (original_img + 1) / 2

        for j, img in enumerate(pred_img):
            plt.imsave(f'./samples/celeba_reconstructions_bs_T={args.time_steps}_N={args.num_samples}/images/image_{count+j}.png', img.cpu().permute([1, 2, 0]).numpy())
        
        # import ipdb; ipdb.set_trace()
        count += len(pred_img)
        # all_imgs.append(pred_img.cpu().numpy())

        # if args.num_samples > 0 and count > args.num_samples: break

    # all_imgs = np.concatenate(all_imgs, axis=0)
    # np.save(f'./samples/reconstruction_T={args.time_steps}_N={args.num_samples}/numpys/all_imgs.npy', all_imgs)