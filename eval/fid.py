import torch.nn as nn
from torchvision import models, transforms
import torch

import numpy as np
from scipy import linalg
from PIL import Image
from pathlib import Path
import os
from typing import List


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path2images: List[str], img_size: int):
        self.path2images = path2images
        self.img_size = img_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), # convert PIL image [H, W, 3] [0, 255] to tensor [3, H, W][0 - 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.path2images)

    def __getitem__(self, i):
        image_path = self.path2images[i]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image


class InceptionV3(nn.Module):
    def __init__(self, for_train=False):
        super().__init__()
        self.for_train = for_train

        inception = models.inception_v3(pretrained=False)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        self.block4 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.final_fc = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        if self.for_train:
            return self.final_fc(x)
        else:
            return x

def get_images_dataloader(img_dir: str,img_size: int, batch_size: int):
    '''
    Get dataloader for images in a directory.
    :param img_dir: path to directory of images
    :param img_size: size of images
    :param batch_size: batch size
    :return: dataloader for images
    '''
    # get all the images in the directory
    path2images = [os.path.join(img_dir, image) for image in os.listdir(img_dir) if
                   image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg')]
    dataset = ImageDataset(path2images, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=4)

    return dataloader

def compute_FID_on_images(path2images_x: str,
                          path2images_y: str,
                          path2afhq_inception_ckpt: str,
                          img_size: int = 100,
                          batch_size: int = 1):
    '''
    Compute FID score between two batches of images.
    :param path2images_x: path to directory for first batch of images
    :param path2images_y: path to directory second batch of images
    '''
    # Choose to evaluate on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    inceptionV3 = InceptionV3()
    inceptionV3.load_state_dict(torch.load(path2afhq_inception_ckpt, map_location="cpu"))
    inceptionV3.eval().to(device)

    # Get dataloader for two set of images
    data_laoder_x = get_images_dataloader(path2images_x, img_size, batch_size)
    data_loader_y = get_images_dataloader(path2images_y, img_size, batch_size)

    feature_x_list = []
    for xs in data_laoder_x:
        xs = xs.to(device)
        feature_x = inceptionV3(xs) # [B, 2048]
        feature_x_list.append(feature_x)

    feature_x = torch.cat(feature_x_list, dim=0).detach().cpu().numpy() # [N, 2048]

    feature_y_list = []

    for ys in data_loader_y:
        ys = ys.to(device)
        feature_y = inceptionV3(ys)
        feature_y_list.append(feature_y)
    feature_y = torch.cat(feature_y_list, dim=0).detach().cpu().numpy() # [N, 2048]

    # Compute FID score
    mean_x = np.mean(feature_x, axis=0) # [2048]
    mean_y = np.mean(feature_y, axis=0) # [2048]

    # set rowvar=False to treat each row as an observation and each column as a variable
    cov_x = np.cov(feature_x, rowvar=False) # [2048, 2048]
    cov_y = np.cov(feature_y, rowvar=False) # [2048, 2048]

    mean_diff = mean_x - mean_y
    cov_sqrt, _= linalg.sqrtm(cov_x @ cov_y, disp=False)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real


    return np.dot(mean_diff, mean_diff) + np.trace(cov_x + cov_y - 2 * cov_sqrt)


if __name__ == '__main__':
    # Example usage python eval/fid.py
    path2images_x = 'datasets/afhq/val/cat'
    path2images_y = 'datasets/afhq/val/dog'
    path2afhq_inception_ckpt = 'eval/afhq_inception_v3.ckpt'

    fid_score = compute_FID_on_images(path2images_x, path2images_y, path2afhq_inception_ckpt)
    print(f'FID score: {fid_score}')
