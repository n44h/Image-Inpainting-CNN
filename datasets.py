"""
Class responsible for pre-processing the dataset.
"""
import os
import numpy as np
from glob import glob
from PIL import Image, ImageStat

import torch
from torch.utils.data import Dataset
from utils import crop_image, generate_input, normalize_image


class RawDataset(Dataset):
    def __init__(self, dir_path: str):
        """ Preparing the raw Dataset """
        # Storing the filepaths of the image samples in image_paths.
        self.image_paths = glob(os.path.join(dir_path, '**', '*.jpg'), recursive=True)

        # Stores the cropped 100x100 images as numpy arrays.
        self.dataset = []

        # Initialize mean and std as 1D torch tensors with float64 datatype.
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)

        # Loop through the list of image paths.
        for path in self.image_paths:
            # Open the image as a PIL image.
            pil_image = Image.open(path)

            # Crop the image to 100x100 and convert to numpy array
            image = crop_image(pil_image, 100, 100)

            # Getting Statistics about the image.
            image_stats = ImageStat.Stat(image)

            # Adding the current image's mean and standard deviation to the respective class attributes.
            self.mean += image_stats.mean
            self.std += image_stats.stddev

            # Convert image to numpy array and add to self.dataset.
            self.dataset.append(np.array(image, dtype=np.float32))

        # Dividing the global mean and std by the number of files to get the average.
        self.mean /= len(self.dataset)
        self.std /= len(self.dataset)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Returning the image file (as numpy array).
        return self.dataset[idx]


class InpaintingDataset(Dataset):
    def __init__(self, dataset: RawDataset):
        """ Generate inputs using the provided RawDataset """
        self.dataset = []
        self.mean = dataset.mean
        self.std = dataset.std

        for sample in dataset:
            # Generate random offsets.
            offset = (np.random.randint(0, 9), np.random.randint(0, 9))

            # Generate random spacings.
            spacing = (np.random.randint(2, 7), np.random.randint(2, 7))

            # Normalize sample.
            sample = normalize_image(sample, mean=self.mean, std=self.std)

            # Generate the input_array, known_array and target array.
            input_array, known_array, target_array = generate_input(image_array=sample,
                                                                    offset=offset,
                                                                    spacing=spacing)

            # Reshape known array from 2D to 3D with just it's first channel layer.
            known_array = torch.reshape(known_array[0, :, :], shape=(1, 100, 100))

            # Concatenate the first layer of known_array to input_array.
            input_array = torch.cat((input_array, known_array), dim=0)

            # Add the generated inputs to self.dataset as a tuple (input, target).
            self.dataset.append((input_array.float(), target_array.float()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Returning the input_array and target_array.
        return self.dataset[idx]
