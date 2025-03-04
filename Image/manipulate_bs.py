from templates import *
from templates_latent import *
from templates_cls import *
from experiment_classifier import ClsModel
import matplotlib.pyplot as plt


device = 'cuda:3'
conf = ffhq128_autoenc_joint()
conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
conf.name = 'eval'
# conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
# state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
# state = torch.load(f'checkpoints/diffae_ckpt/celeba128_eval_70_norm/last.ckpt', map_location='cpu')

model.ema_model.eval()
model.ema_model.to(device)

cls_conf = ffhq128_autoenc_cls()
cls_model = ClsModel(cls_conf)
state = torch.load(f'checkpoints/diffae_ckpt/{cls_conf.name}/last.ckpt',
                    map_location='cpu')
print('latent step:', state['global_step'])
cls_model.load_state_dict(state['state_dict'], strict=False);
cls_model.to(device);

data = CelebHQAttrDataset(data_paths['celebahq'],
                                    conf.img_size,
                                    data_paths['celebahq_anno'],
                                    do_augment=False, mode='test')

batch = data[1]['img'].unsqueeze(0)

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250)

cls_id = CelebHQAttrDataset.cls_to_id['Smiling']

cond2 = cls_model.normalize(cond)
cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
cond2 = cls_model.denormalize(cond2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# img = model.render(xT, cond2, T=100)
sampler = conf._make_diffusion_conf(100).make_sampler()
img = sampler.sample(model=model.ema_model, noise=xT, cond=cond2)
img = (img + 1) / 2

ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(img[0].permute(1, 2, 0).cpu())
plt.savefig('imgs_manipulated/compare_bs.png')




