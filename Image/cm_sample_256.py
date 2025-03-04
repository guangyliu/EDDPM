import os
import torch
import numpy as np
from PIL import Image
from diffusers import ConsistencyModelPipeline
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = 'cuda'
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_bedroom256_lpips"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

n_samples = 50000
batch_size = 10

logdir = './samples/consistency_model_twosteps/samples_lpips_256'
if not os.path.exists(logdir):
    os.makedirs(logdir)
    for i in range(n_samples // batch_size):
        images = pipe(timesteps=[18, 0], batch_size=batch_size)
        for j in range(batch_size):
            image = images.images[j]
            image.save(f'{logdir}/image_{i*batch_size+j}.png')
