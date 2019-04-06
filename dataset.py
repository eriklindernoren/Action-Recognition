import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(self, dataset_path, indices, input_shape, sequence_length, training=True):
        channels, height, width = input_shape
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.sequence_length = sequence_length
        self.sequences = sorted(
            [
                seq_path
                for i, seq_path in enumerate(glob.glob(f"{dataset_path}/*/*"))
                if len(os.listdir(seq_path)) >= sequence_length and i in indices
            ]
        )
        self.labels = sorted(list(set([self._label_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.labels)
        self.training = training

    def _label_from_path(self, path):
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort sequence frames based on frame number
        frames = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=lambda path: self._frame_number(path))
        # Start index and sample interval for the test set
        start_i, sample_interval = 0, len(frames) // self.sequence_length
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(frames) // self.sequence_length + 1)
            start_i = np.random.randint(0, len(frames) - sample_interval * self.sequence_length + 1)
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(frames), sample_interval):
            image_path = frames[i]
            if len(image_sequence) < self.sequence_length:
                image_sequence.append(self.transform(Image.open(image_path)))
        image_sequence = torch.stack(image_sequence)
        label = self._label_from_path(sequence_path)
        return image_sequence, self.labels.index(label)

    def __len__(self):
        return len(self.sequences)
