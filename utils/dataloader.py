import numpy as np
import tonic
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_dataloaders(data_dir, batch_size, num_workers, T=10, subset_ratio=0.6):
    # Define transforms
    transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(time_factor=10, spatial_factor=2),
        tonic.transforms.ToFrame(sensor_size=(2, 34, 34), time_window=1)
    ])

    # Load datasets
    train_set = NMNIST(root=data_dir, train=True, data_type='frame', frames_number=T, split_by='number')
    test_set = NMNIST(root=data_dir, train=False, data_type='frame', frames_number=T, split_by='number')

    train_loader = DataLoader( train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=True    )
    test_loader = DataLoader( test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False    )

    return train_loader, test_loader
    
    
'''
    # Subset for quick training/testing
    train_size = int(subset_ratio * len(train_set))
    test_size = int(subset_ratio * len(test_set))
    train_indices = np.random.choice(len(train_set), size=train_size, replace=False)
    test_indices = np.random.choice(len(test_set), size=test_size, replace=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
'''

