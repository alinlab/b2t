import os
import clip
import torch
from pathlib import Path
import skimage.io as io
from PIL import Image
from tqdm import tqdm

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def calc_similarity(image_dir, images, keywords):
    # Load the model
    images = [image_dir + image for image in images]
    images = [Image.fromarray(io.imread(image)) for image in images]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    similarity_list = []
    image_list_chunked = list_chunk(images, 2000)

    for image_list in tqdm(image_list_chunked):


        # Prepare the inputs
        image_inputs = torch.cat([preprocess(pil_image).unsqueeze(0) for pil_image in image_list]).to(device) # (1909, 3, 224, 224)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in keywords]).to(device)


        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T) # (1909, 20)
        similarity_list.append(similarity)

    similarity = torch.cat(similarity_list).mean(dim=0)

    return similarity