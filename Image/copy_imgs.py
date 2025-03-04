from templates import *
from templates_latent import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FFHQlmdb
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose


import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=420, type=int)
parser.add_argument("--device", default=0)
parser.add_argument('--dataset', default='celeba', type=str)

args = parser.parse_args()

with torch.no_grad():
    device = f'cuda:{args.device}'
    if args.dataset == 'celeba':
        conf = celeba64d2c_autoenc_joint()
    elif args.dataset == 'ffhq':
        conf = ffhq128_autoenc_joint()
    elif args.dataset == 'horse':
        conf = horse128_autoenc_joint()
    elif args.dataset == 'bedroom':
        conf = bedroom128_autoenc_joint()
    conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
    conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
    conf.T_eval = 100
    conf.latent_T_eval = 100
    conf.name = 'eval'
    dataset = conf.make_dataset()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # print(conf.name)
    torch.manual_seed(4)

    os.makedirs(f'./samples/{args.dataset}/original_imgs', exist_ok=True)

    count = 0
    # all_imgs = []
    for batch in tqdm(dataloader):

        original_img = batch['img']
        original_img = (original_img + 1) / 2

        for j in range(len(original_img)):
            if not os.path.exists(f'./samples/{args.dataset}/original_imgs/original_image_{count+j}.png'):
                plt.imsave(f'./samples/{args.dataset}/original_imgs/original_image_{count+j}.png', original_img[j].cpu().permute([1, 2, 0]).numpy())
        
        count += len(original_img)
        
        if count > 50000: break