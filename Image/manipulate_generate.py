from templates import *
from templates_latent import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import random

device = 'cuda:3'

# load our model
conf = ffhq128_autoenc_joint()
conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
conf.name = 'eval'
# conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
# state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
# state = torch.load(f'checkpoints/diffae_ckpt/celeba128_eval_70_norm/last.ckpt', map_location='cpu')
model.model = torch.load(f'./checkpoints/diffae_ckpt/v3/v6_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt', map_location='cpu')
model.ema_model = model.model
model.ema_model.eval()
model.ema_model.to(device)

cls_conf = ffhq128_autoenc_joint_cls()
cls_conf.name = 'celeba128_eval_70_norm'
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/diffae_ckpt/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False);
cls_model.to(device);

sampler = conf._make_diffusion_conf(100).make_sampler()


# load baseline model
conf = ffhq128_autoenc_joint()
conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
conf.name = 'eval'
# conf = ffhq256_autoenc()
# print(conf.name)
model_bs = LitModel(conf)
# state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
# state = torch.load(f'checkpoints/diffae_ckpt/celeba128_eval_70_norm/last.ckpt', map_location='cpu')

model_bs.ema_model.eval()
model_bs.ema_model.to(device)

cls_conf = ffhq128_autoenc_cls()
cls_model_bs = ClsModel(cls_conf)
state = torch.load(f'checkpoints/diffae_ckpt/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model_bs.load_state_dict(state['state_dict'], strict=False);
cls_model_bs.to(device);

sampler_bs = conf._make_diffusion_conf(100).make_sampler()


data = CelebHQAttrDataset(data_paths['celebahq'],
                                    conf.img_size,
                                    data_paths['celebahq_anno'],
                                    do_augment=False, mode='test')

import ipdb; ipdb.set_trace()

random.seed(0)
indices = random.sample(range(len(data)), 5000)[:1250]
data = Subset(data, indices)

os.makedirs(f'./manipulate/bs_0.1_2', exist_ok=True)
os.makedirs(f'./manipulate/ori_0.1_2', exist_ok=True)
os.makedirs(f'./manipulate/ours_0.1_2', exist_ok=True)
os.makedirs(f'./manipulate/bs_0.3_2', exist_ok=True)
os.makedirs(f'./manipulate/ori_0.3_2', exist_ok=True)
os.makedirs(f'./manipulate/ours_0.3_2', exist_ok=True)

dataloader = DataLoader(data, batch_size=200, shuffle=False)

count = 0

with torch.no_grad():

    for batch in dataloader:
        
        batch = batch['img']
        
        for cls_name in tqdm(CelebHQAttrDataset.cls_to_id, total=len(CelebHQAttrDataset.cls_to_id)):
            
            for extent in [0.1, 0.3]:
            
                cls_id = CelebHQAttrDataset.cls_to_id[cls_name]
                
                cond = model.encode(batch.to(device))
                xT = model.encode_stochastic(batch.to(device), cond, T=250)
                cond2 = cls_model.normalize(cond)
                cond2 = cond2 + extent * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
                cond2 = cls_model.denormalize(cond2)
                
                cond_bs = model_bs.encode(batch.to(device))
                xT_bs = model_bs.encode_stochastic(batch.to(device), cond, T=250)
                cond2_bs = cls_model_bs.normalize(cond)
                cond2_bs = cond2_bs + extent * math.sqrt(512) * F.normalize(cls_model_bs.classifier.weight[cls_id][None, :], dim=1)
                cond2_bs = cls_model_bs.denormalize(cond2_bs)

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                # img = model.render(xT, cond2, T=100)
                img_bs = sampler_bs.sample(model=model_bs.ema_model, noise=xT_bs, cond=cond2_bs)
                img_bs = (img_bs + 1) / 2

                img = sampler.sample(model=model.model, noise=xT, cond=cond2)
                img = (img + 1) / 2

                ori = (batch + 1) / 2
                
                for j in range(len(img)):
                    plt.imsave(f"./manipulate/bs_{extent}/{cls_name}_{count + j}.png", img_bs[j].cpu().permute([1, 2, 0]).numpy())
                    plt.imsave(f"./manipulate/ours_{extent}/{cls_name}_{count + j}.png", img[j].cpu().permute([1, 2, 0]).numpy())
                    plt.imsave(f"./manipulate/ori_{extent}/{cls_name}_{count + j}.png", ori[j].cpu().permute([1, 2, 0]).numpy())
                    
        count += len(batch)

        # ori = (batch + 1) / 2
        # ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
        # ax[1].imshow(img_bs[0].permute(1, 2, 0).cpu())
        # ax[2].imshow(img[0].permute(1, 2, 0).cpu())
        # plt.savefig(f'imgs_manipulated/compare_{cls_name}_{i}_{extent}.png')
