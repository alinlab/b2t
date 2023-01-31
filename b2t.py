import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import clip

# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.waterbirds import Waterbirds, get_transform_cub

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity

from tqdm import tqdm
import os
import time
import pandas as pd


import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type = str, default = 'waterbird', help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_CUB_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_CUB_erm.pth, best_model_CUB_dro.pth
    parser.add_argument("--extract_caption", default = True)
    parser.add_argument("--save_result", default = True)
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
args = parse_args()

if args.dataset == 'waterbird':
    preprocess = get_transform_cub()
    class_names = ['landbird', 'waterbird']
    # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
    image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
    caption_dir = 'data/cub/caption/'
    val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)
elif args.dataset == 'celeba':
    preprocess = get_transform_celeba()
    class_names = ['not blond', 'blond']
    # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
    image_dir = 'data/celebA/data/img_align_celeba/'
    caption_dir = 'data/celebA/caption/'
    val_dataset = CelebA(data_dir='data/celebA/data/', split='val', transform=preprocess)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)




result_dir = 'result/'
model_dir = 'model/'
diff_dir = 'diff/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(diff_dir):
    os.makedirs(diff_dir)

# extract caption
if args.extract_caption:
    print("Start extracting captions..")
    for x, (y, y_group, y_spurious), idx, path in tqdm(val_dataset):
        image_path = image_dir + path
        caption = extract_caption(image_path)
        if not os.path.exists(caption_dir):
            os.makedirs(caption_dir)
        caption_path = caption_dir + path.split("/")[-1][:-4] + ".txt"
        with open(caption_path, 'w') as f:
            f.write(caption)
    print("Captions of {} images extracted".format(len(val_dataset)))

# correctify dataset
result_path = result_dir + args.dataset +"_" +  args.model.split(".")[0] + ".csv"
if not os.path.exists(result_path):
    model = torch.load(model_dir + args.model)
    model = model.to(device)
    model.eval()
    start_time = time.time()
    print("Pretrained model \"{}\" loaded".format(args.model))

    result = {"image":[],
            "pred":[],
            "actual":[],
            "group":[],
            "spurious":[],                
            "correct":[],
            "caption":[],
            }

    with torch.no_grad():
        running_corrects = 0
        for (images, (targets, targets_g, targets_s), index, paths) in tqdm(val_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                image = paths[i]
                pred = preds[i]
                actual = targets[i]
                group = targets_g[i]
                spurious = targets_s[i]
                caption_path = caption_dir + image.split("/")[-1][:-4] + ".txt"
                with open(caption_path, "r") as f:
                    caption = f.readline()
                result['image'].append(image)
                result['pred'].append(pred.item())
                result['actual'].append(actual.item())
                result['group'].append(group.item())
                result['spurious'].append(spurious.item())
                result['caption'].append(caption)
                if pred == actual:
                        result['correct'].append(1)
                        running_corrects += 1
                else:
                        result['correct'].append(0)

        print("# of correct examples : ", running_corrects)
        print("# of wrong examples : ", len(val_dataset) - running_corrects)
        print("# of all examples : ", len(val_dataset))
        print("Accuracy : {:.2f} %".format(running_corrects/len(val_dataset)*100))

    df = pd.DataFrame(result)
    df.to_csv(result_path)
    print("Classified result stored")
else:
    df = pd.read_csv(result_path)
    print("Classified result \"{}\" loaded".format(result_path))

# extract keyword
df_wrong = df[df['correct'] == 0]
df_correct = df[df['correct'] == 1]
df_class_0 = df[df['actual'] == 0] # not blond, landbird
df_class_1 = df[df['actual'] == 1] # blond, waterbird
df_wrong_class_0 = df_wrong[df_wrong['actual'] == 0]
df_wrong_class_1 = df_wrong[df_wrong['actual'] == 1]
df_correct_class_0 = df_correct[df_correct['actual'] == 0]
df_correct_class_1 = df_correct[df_correct['actual'] == 1]

caption_wrong_class_0 = ' '.join(df_wrong_class_0['caption'].tolist())
caption_wrong_class_1 = ' '.join(df_wrong_class_1['caption'].tolist())

keywords_class_0 = extract_keyword(caption_wrong_class_0)
keywords_class_1 = extract_keyword(caption_wrong_class_1)

# calculate similarity
print("Start calculating scores..")
similarity_wrong_class_0 = calc_similarity(image_dir, df_wrong_class_0['image'], keywords_class_0)
similarity_correct_class_0 = calc_similarity(image_dir, df_correct_class_0['image'], keywords_class_0)
similarity_wrong_class_1 = calc_similarity(image_dir, df_wrong_class_1['image'], keywords_class_1)
similarity_correct_class_1 = calc_similarity(image_dir, df_correct_class_1['image'], keywords_class_1)

dist_class_0 = similarity_wrong_class_0 - similarity_correct_class_0
dist_class_1 = similarity_wrong_class_1 - similarity_correct_class_1

print("Result for class :", class_names[0])
diff_0 = print_similarity(keywords_class_0, keywords_class_1, dist_class_0, dist_class_1, df_class_0)
print("*"*60)
print("Result for class :", class_names[1])
diff_1 = print_similarity(keywords_class_1, keywords_class_0, dist_class_1, dist_class_0, df_class_1)

if args.save_result:
    diff_path_0 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[0] + ".csv"
    diff_path_1 = diff_dir + args.dataset +"_" +  args.model.split(".")[0] + "_" +  class_names[1] + ".csv"
    diff_0.to_csv(diff_path_0)
    diff_1.to_csv(diff_path_1)