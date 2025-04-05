import os
import urllib.request
import zipfile
from  multiprocessing.pool import Pool
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image


class AFHQDataset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 dataset_split: str,
                 max_samples_per_class: int = -1,
                 class_label_offset: int = 1, # 0 for NULL class
                 transform=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_split = dataset_split
        self.max_samples_per_class = max_samples_per_class
        self.class_label_offset = class_label_offset
        self.transform = transform

        # e.g. cat images are udner data/afhq/train/cat
        self.categories = os.listdir(self.root_dir / self.dataset_split)

        # print(f'categories {self.categories}')

        self.num_categories = len(self.categories)
        self.catID2label = {(i + self.class_label_offset) : cat for i, cat in
                            enumerate(self.categories)}

        self.path2imgs = []
        self.labels = []

        # Load all images and labels by category
        for catID, cat in enumerate(self.categories):
            cat_dir = self.root_dir / self.dataset_split / cat

            # print(f'image cat path: {cat_dir}')

            img_paths = self._list_images(cat_dir)
            if self.max_samples_per_class > 0:
                img_paths = img_paths[:self.max_samples_per_class]
            self.path2imgs += sorted(img_paths)
            self.labels.extend([catID + self.class_label_offset] * len(img_paths))

    def _list_images(self, imgs_path: Path):
        valid_exts = {".png", ".jpg", ".jpeg"}
        return [
            f for f in Path(imgs_path).rglob("*") # rglob means recursively search for all files
            if f.suffix.lower() in valid_exts
        ]


    def __getitem__(self, idx):
        img_path = self.path2imgs[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB") # shape: (H, W, 3)

        if self.transform is not None:
            # torchvision.transforms expects a PIL Image or numpy array of shape (H, W, C) in
            # the range [0, 255]
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.path2imgs)


class AFHQDatasetHelper():
    '''
    Helper class to download and create a DataLoader for the AFHQ dataset.
    '''
    def __init__(self,
                 root_dir: str,
                 img_resolution: int = 64,
                 max_samples_per_class: int = -1,
                 class_label_offset: int = 1, # 0 for NULL class
                 batch_size: int = 32,
                 transform: bool = None,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle: bool = True,
                 drop_last: bool = False,
    ):
        self.root_dir = root_dir
        self.resolution = img_resolution
        self.max_samples_per_class = max_samples_per_class
        self.class_label_offset = class_label_offset
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Set the transform for the dataset
        # Resize to 64x64 and normalize to [-1, 1]
        # x -0.5 / 0.5 lies in the range [-1, 1] if x lies in the range [0, 1]
        # Normalize to [-1, 1] is needed for DDPM
        '''
        torchvision.transforms.ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
        torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to
        one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or
        if the numpy.ndarray has dtype = np.uint8

        Normalize a tensor image with mean and standard deviation. This transform does not support
        PIL Image. Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
        this transform will normalize each channel of the input torch.*Tensor i.e.,
        output[channel] = (input[channel] - mean[channel]) / std[channel]
        '''
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.resolution, self.resolution)), # Resize to 64x64
                    transforms.ToTensor(), # Convert to ToTensor and normalize to [0, 1]
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
                ]
            )


        # doiwnload the dataset if it doesn't exist
        if not os.path.exists(self.root_dir):
            self._download_dataset()

        self.afhq_root = self.root_dir / Path("afhq")

        # Set train and val set for DataLoaders
        self._create_datasets()


    def _download_dataset(self):

        # For dropbox, dl=1 is required to download
        url = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
        zip_path = os.path.join(self.root_dir, "afhq.zip")

        print(f'zip_path: {zip_path}')

        #  exist_ok = True means “Create the directory (and any parent directories), but don’t raise an error if it already exists.”
        os.makedirs(self.root_dir, exist_ok=True)

        print(f'Downloading AFHQ Dataset from {url}...')
        urllib.request.urlretrieve(url, zip_path)
        print('Download is complete.')

        print("Extracting ZIP file ...")
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        else:
            print("The downloaded file is not a valid ZIP file.")
            return
        print("Extraction complete!")

        os.remove(zip_path)
        print("Cleaned up temporary ZIP.")

    def _create_datasets(self):
        # Create the dataset
        self.train_set = AFHQDataset(
            root_dir=self.afhq_root,
            dataset_split="train",
            max_samples_per_class=self.max_samples_per_class,
            class_label_offset=self.class_label_offset,
            transform=self.transform
        )

        self.val_set = AFHQDataset(
            root_dir=self.afhq_root,
            dataset_split="val",
            max_samples_per_class=self.max_samples_per_class,
            class_label_offset=self.class_label_offset,
            transform=self.transform
        )

        self.num_classes = self.train_set.num_categories

    def get_train_loader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

        return train_loader

    def get_val_loader(self):
        # Create the DataLoader
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

        return val_loader

#
# if __name__ == "__main__":
#     # Test wheter dataset is downloaded and DataLoader is working
#     root_dir = "data"
#     img_resolution = 64
#     max_samples_per_class = 1000000
#     batch_size = 32
#     num_workers = 4
#     pin_memory = True
#     shuffle = True
#     drop_last = True
#
#     dataset_helper = AFHQDatasetHelper(
#         root_dir=root_dir,
#         img_resolution=img_resolution,
#         max_samples_per_class=max_samples_per_class,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=shuffle,
#         drop_last=drop_last
#     )
#
#     train_loader = dataset_helper.get_train_loader()
#     val_loader = dataset_helper.get_val_loader()
#     print(f"Number of classes: {dataset_helper.num_classes}")
#     print(f"Number of train samples: {len(dataset_helper.train_set)}")
#     print(f"Number of val samples: {len(dataset_helper.val_set)}")
#
#     for i, (img, label) in enumerate(train_loader):
#         print(f"Batch {i}:")
#         print(f"Image shape: {img.shape}")
#         print(f"Label shape: {label.shape}")
#         print(f"Label: {label}")
