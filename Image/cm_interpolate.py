import torch

from diffusers import ConsistencyModelPipeline, image_processor

device = "cuda"
# Load the cd_imagenet64_l2 checkpoint.
model_id_or_path = "openai/diffusers-cd_bedroom256_lpips"
pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.to(device)

import os
from PIL import Image
import numpy as np


for ratio in [0.0, 0.2, 0.4]:
        
    os.makedirs(f"samples/consistency_model_twosteps/interpolation/{ratio}")

    for i in range(50000):

        file1, file2 = os.listdir("samples/bedroom/original_imgs_256")[i:i+2]
        batch_a = Image.open(f"samples/bedroom/original_imgs_256/{file1}").convert("RGB")
        batch_b = Image.open(f"samples/bedroom/original_imgs_256/{file2}").convert("RGB")

        batch_a = torch.tensor(np.array(batch_a) / 255).permute(2, 0, 1).cuda().to(torch.float16)
        batch_b = torch.tensor(np.array(batch_b) / 255).permute(2, 0, 1).cuda().to(torch.float16)

        batch_a = (batch_a - 0.5) * 2
        batch_b = (batch_b - 0.5) * 2
        batch_a = batch_a.unsqueeze(0)
        batch_b = batch_b.unsqueeze(0)

        pipe.scheduler.set_timesteps(timesteps=[17,0], device=device)
        pipe.scheduler._init_step_index(pipe.scheduler.timesteps[-1])

        sigmas = pipe.scheduler.sigmas

        batch_a = batch_a + sigmas[1] * torch.randn_like(batch_a)
        batch_b = batch_b + sigmas[1] * torch.randn_like(batch_a)

        # new_batch = batch_a * ratio + batch_b * (1 - ratio)
        new_batch = batch_a

        scaled_sample = pipe.scheduler.scale_model_input(new_batch, pipe.scheduler.timesteps[-1])
        model_output = pipe.unet(scaled_sample, pipe.scheduler.timesteps[-1], return_dict=False)[0]
        sample = pipe.scheduler.step(model_output, pipe.scheduler.timesteps[-1], new_batch, generator=None)[0]

        pipe.postprocess_image(sample.detach(), output_type='pil')[0].resize((128, 128)).save("samples/consistency_model_twosteps/interpolation/{ratio}/{i}.png".format(ratio=ratio, i=i))