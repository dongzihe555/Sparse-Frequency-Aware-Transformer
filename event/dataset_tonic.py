
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

def get_tonic_datasets():
    try:
        import tonic
        import tonic.transforms as transforms
        return tonic, transforms
    except ImportError as e:
        raise ImportError(
            "tonic is not installed. Please install it first:\n"
            "  pip install tonic"
        ) from e

class TonicCIFAR10DVS(Dataset):
    def __init__(self, root, train=True, T=16):
        self.T = T
        self.train = train
        tonic, transforms = get_tonic_datasets()
        
        self.sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
        self.frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=self.sensor_size, n_time_bins=T),
        ])
        
        tonic_root = os.path.join(root, 'tonic')
        os.makedirs(tonic_root, exist_ok=True)
        
        self.dataset = tonic.datasets.CIFAR10DVS(
            save_to=tonic_root,
            transform=self.frame_transform
        )
        
        self.indices = self._split_indices()
    
    def _split_indices(self):

        targets = np.array([self.dataset[i][1] for i in range(len(self.dataset))])
        indices = []
        for c in range(10):
            class_idx = np.where(targets == c)[0]
            n_train = int(len(class_idx) * 0.9)
            if self.train:
                indices.extend(class_idx[:n_train])
            else:
                indices.extend(class_idx[n_train:])
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        frames, label = self.dataset[real_idx]

        frames = torch.from_numpy(frames).float()
        return frames, label

class TonicDVSGesture(Dataset):
    def __init__(self, root, train=True, T=16):
        self.T = T
        self.train = train
        tonic, transforms = get_tonic_datasets()
        
        self.sensor_size = tonic.datasets.DVSGesture.sensor_size
        self.frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=self.sensor_size, n_time_bins=T),
        ])
        
        tonic_root = os.path.join(root, 'tonic')
        os.makedirs(tonic_root, exist_ok=True)
        
        self.dataset = tonic.datasets.DVSGesture(
            save_to=tonic_root,
            train=train,
            transform=self.frame_transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        frames, label = self.dataset[idx]
        frames = torch.from_numpy(frames).float()
        return frames, label

def load_data_tonic(dataset, dataset_dir, distributed, T):
    import torch.utils.data.distributed as dist_data
    
    print("Loading data (tonic backend)")
    
    if dataset == 'cifar10dvs':
        dataset_root = os.path.join(dataset_dir, 'CIFAR10DVS')
        dataset_train = TonicCIFAR10DVS(root=dataset_root, train=True, T=T)
        dataset_test = TonicCIFAR10DVS(root=dataset_root, train=False, T=T)
    elif dataset == 'dvsgesture':
        dataset_root = os.path.join(dataset_dir, 'DVS128Gesture')
        dataset_train = TonicDVSGesture(root=dataset_root, train=True, T=T)
        dataset_test = TonicDVSGesture(root=dataset_root, train=False, T=T)
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported by tonic backend.")
    
    if distributed:
        train_sampler = dist_data.DistributedSampler(dataset_train)
        test_sampler = dist_data.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    return dataset_train, dataset_test, train_sampler, test_sampler
