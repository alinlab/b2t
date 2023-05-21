import os
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

from custom_model import CustomSemanticEditPipeline


# load diffusion model
device = 'cuda:0'
seed = 0

pipe = CustomSemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

# generate images
prompt = "a photo of a nurse."
num_images = 100

gen = torch.Generator(device=device)
transform = T.ToTensor()

gen.manual_seed(seed)
for i in tqdm(range(num_images)):
    out = pipe(prompt=prompt, generator=gen)
    images = out.images[0]
    images.save("images/{}.png".format(i))

# load generated images
images = []
for i in range(num_images):
    image = Image.open(os.path.join(save_directory, "{}.png".format(i)))
    images.append(transform(image))
images = torch.stack(images).to(device)
images = images * 2 - 1  # range: (-1,1)

# compute SD score
keyword_prompts = [
    "woman",
    "stethoscope",
]

base_noise = pipe.pred_noise(images=images, prompt=prompt)
for p in keyword_prompts:
    noise = pipe.pred_noise(images=images, prompt=p)
    score = (noise - base_noise).flatten(start_dim=1).norm(dim=1)
    print(prompt, "{:.4f}".format(score.mean().item()))

# generate fair images
edit = "stethoscope"

gen.manual_seed(seed)
for i in tqdm(range(num_images)):
    params_edit = {'editing_prompt': edit,
                   'reverse_editing_direction': True,
                   'edit_warmup_steps': 5,
                   'edit_guidance_scale': 4,
                   'edit_threshold': 0.95,
                   'edit_momentum_scale': 0.5,
                   'edit_mom_beta': 0.6}

    out = pipe(**params_edit, generator=gen)
    image = out.images[0]
    images.save("fair_images/{}.png".format(i))
