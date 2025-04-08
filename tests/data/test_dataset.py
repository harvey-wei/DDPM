import unittest
from pathlib import Path
import sys
import os

# Allow imports from the project root
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.dataset import AFHQDatasetHelper  # now correctly imports your dataset
from utils.get_infinite_generator import get_infinite_generator

class TestAFHQDataset(unittest.TestCase):
    def setUp(self):
        self.root_dir = "datasets"
        self.img_resolution = 64
        self.max_samples = 100
        self.batch_size = 16
        self.num_workers = 4
        self.shuffle = True
        self.drop_last = True

        self.dataset_helper = AFHQDatasetHelper(
            root_dir=self.root_dir,
            img_resolution=self.img_resolution,
            max_samples_per_class=self.max_samples,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataset_loaded(self):
        self.assertGreater(len(self.dataset_helper.train_set), 0)
        self.assertGreater(len(self.dataset_helper.val_set), 0)

    def test_loader_shapes(self):
        train_loader = self.dataset_helper.get_train_loader()
        train_loader_gen =get_infinite_generator(train_loader)
        # batch = next(iter(train_loader))
        batch = next(train_loader_gen)
        batch = next(train_loader_gen)
        images, labels = batch

        self.assertEqual(images.shape[1:], (3, self.img_resolution, self.img_resolution))
        self.assertEqual(images.shape[0], self.batch_size)
        self.assertEqual(labels.shape[0], self.batch_size)

    def test_loader_img_pixel_range(self):
        train_loader = self.dataset_helper.get_train_loader()
        batch = next(iter(train_loader))
        images, _ = batch

        # Check if pixel values are in the range [-1, 1]
        self.assertTrue(((images >= -1.0) & (images <= 1.0)).all().item())

    def test_num_classes_positive(self):
        self.assertGreater(self.dataset_helper.num_classes, 0)


if __name__ == "__main__":
    unittest.main()
