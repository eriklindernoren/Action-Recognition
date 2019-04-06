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
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length=40, training=True):
        self.training = training
        self.label_mapping = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length
        keep_i = [i for i, seq_path in enumerate(self.sequences) if len(os.listdir(seq_path)) >= sequence_length]
        self.sequences = [x for i, x in enumerate(self.sequences) if i in keep_i]
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label)
        return label_mapping

    def _extract_sequence_paths(
        self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True
    ):
        assert split_number in [1, 2, 3], "Split number has to be one of {1, 2, 3}"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            seq_path = line.split()[0]
            sequence_paths += [os.path.join(dataset_path, seq_path.split(".avi")[0])]
            self
        return sequence_paths

    def _activity_from_path(self, path):
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
        target = self.label_mapping[self._activity_from_path(sequence_path)] - 1
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
