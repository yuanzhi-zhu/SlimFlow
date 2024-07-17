import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os
import logging
join = os.path.join

extensions = ['.jpg', '.jpeg', '.JPEG', '.png', '.bmp']

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset(config, phase="train"):
    if config.data.reflow_data_root:
        dataset = PairedDataDataset(config.data.reflow_data_root)
        dataloader = DataLoader(dataset, 
                                 batch_size=config.training.batch_size, 
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)
    else:
        transform_list = [torchvision.transforms.Resize(config.data.image_size)]
        if config.data.random_flip:
            transform_list.append(torchvision.transforms.RandomHorizontalFlip())
        transform_list.extend([
            torchvision.transforms.Grayscale(num_output_channels=1) if config.data.num_channels == 1 else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if config.data.num_channels == 3 else torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        if config.data.dataset == 'CIFAR10':
            dataset = torchvision.datasets.CIFAR10(
                root='datasets/cifar',
                download=True,
                train=True if phase == 'train' else False,
                transform=torchvision.transforms.Compose(transform_list)
            )
            # CIFAR10 class labels
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            # Filter the dataset to only keep 'cat' images
            # if config.sampling has attribute train_subset, then only keep that class
            if hasattr(config.sampling, 'train_subset') and config.sampling.train_subset in classes:
                class_idx = classes.index(config.sampling.train_subset)
                dataset = [(img, label) for img, label in dataset if label == class_idx]
        elif config.data.dataset == 'MNIST':
            dataset = torchvision.datasets.MNIST(
                root='datasets/mnist',
                download=True,
                train=True if phase == 'train' else False,
                transform=torchvision.transforms.Compose(transform_list)
            )
        elif config.data.dataset == 'custom':
            dataset = CustomImageDataset(config.data.custom_data_root, 
                                         transform=torchvision.transforms.Compose(transform_list),
                                         phase=phase
                                         )
        else:
            # TODO: add other datasets
            raise NotImplementedError
        dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=config.training.batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=True)
    return dataloader


class PairedDataDataset(Dataset):
    """Dataset for paired data (z0, z1), loaded from data_root."""
    def __init__(self, data_root):
        z0_cllt = []
        data_cllt = []
        label_cllt = []
        folder_list = os.listdir(data_root)
        for folder in folder_list:
            logging.info(f'FOLDER: {folder}')
            z0 = np.load(os.path.join(data_root, folder, 'z0.npy'))
            logging.info('Loaded z0')
            data = np.load(os.path.join(data_root, folder, 'z1.npy'))
            logging.info('Loaded z1')
            z0 = torch.from_numpy(z0).cpu()
            data = torch.from_numpy(data).cpu()
            z0_cllt.append(z0)
            data_cllt.append(data)
            # load labels if available
            if os.path.exists(os.path.join(data_root, folder, 'label.npy')):
                label = np.load(os.path.join(data_root, folder, 'label.npy'))
                label = torch.from_numpy(label).cpu()
                logging.info(f'Loaded label in shape: {label.shape}')
                label_cllt.append(label)
        logging.info(f'Successfully Loaded (z0, z1) pairs from {data_root}!!!')
        self.z0_cllt = torch.cat(z0_cllt).to(torch.float32)
        self.data_cllt = torch.cat(data_cllt).to(torch.float32)
        if len(label_cllt) > 0:
            self.label_cllt = torch.cat(label_cllt).to(torch.long)
        else:
            self.label_cllt = None
        assert len(self.z0_cllt) == len(self.data_cllt), "Data files have different lengths"

    def __len__(self):
        return len(self.z0_cllt)

    def __getitem__(self, idx):
        z1_sample = self.data_cllt[idx]
        z0_sample = self.z0_cllt[idx]
        # data pair 
        batch = torch.stack([z1_sample, z0_sample], dim=0)
        if self.label_cllt is not None:
            label = self.label_cllt[idx]
        else:
            # dummy label.
            label = torch.zeros((2,), dtype=torch.long)
        return batch, label


class CustomImageDataset(Dataset):
    def __init__(self, data_root, transform=None, phase='train'):
        self.directory = data_root
        self.transform = transform
        self.image_names = [f for f in os.listdir(data_root) if any(f.endswith(ext) for ext in extensions)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        # dummy label
        label = torch.zeros((1,), dtype=torch.long)
        return image, label
