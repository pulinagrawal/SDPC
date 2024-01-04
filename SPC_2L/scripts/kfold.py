from pyexpat import model
import random
import sys
from pathlib import Path
from py import test
import torch
import numpy as np
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Compose, Resize
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle

sys.path.append(str((Path(".").resolve().absolute())))

# Import your modules here
from SPC_2L.DataTools import DataBase, gaussian_kernel, LCN, whitening, z_score, mask, to_device, norm
from SPC_2L.Network import LayerPC, Network
from SPC_2L.Coding import ML_Lasso, ML_FISTA
from SPC_2L.Optimizers import mySGD, myAdam
from SPC_2L.Monitor import Monitor
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

from STL_training import train_STL_model, test_STL_model, setup_model, imshow
import STL_training as stl

# Define your existing functions here

def setup_dataset(data_path, batch_size=64, n_splits=5, seed=42, split='train'):
    """
    Set up the dataset for k-fold cross validation.
    """
    transform = Compose([ToTensor(),
                         to_device(),
                         whitening((96, 96), f_0=0.5),
                         Resize((96, 96)),
                         LCN(kernel_size=9, sigma=0.5, rgb=True),
                         z_score(),
                         mask((96, 96))])

    full_dataset = STL10(data_path, transform=transform, download=True, split='unlabeled')

    #kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_size = 5000*5
    test_size = 5000*5
    dataset_folds = []
    train_dataset = Subset(full_dataset, range(train_size))
    train_indexes = np.linspace(0, len(train_dataset), 6, dtype=int)
    train_indexes = [range(i,j) for i,j in zip(train_indexes[:-1], train_indexes[1:])]

    test_dataset = Subset(full_dataset, range(train_size,train_size+test_size))
    test_indexes = np.linspace(0, len(test_dataset), 6, dtype=int)
    test_indexes = [range(i,j) for i,j in zip(test_indexes[:-1], test_indexes[1:])]

    print(train_indexes, test_indexes)

    for train_index, test_index in zip(train_indexes, test_indexes):
        if split == 'train':
            subset = Subset(train_dataset, train_index)
        else:
            subset = Subset(test_dataset, test_index)

        # Assuming you want to use the DataLoader as in the original setup_dataset function
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Gaussian masks for the dictionaries, assuming it's the same for all folds
        mask_g = [gaussian_kernel((64,3,8,8), sigma=30), gaussian_kernel((128,64,8,8), sigma=30)]

        dataset_folds.append((data_loader, mask_g))

    return dataset_folds


# K-fold cross-validation function remains the same
def k_fold_cross_validation(n_splits, nb_epoch, batch_size, data_path):
    """
    Perform k-fold cross validation
    """
    train_datasets = setup_dataset(data_path, split='train', batch_size=batch_size, n_splits=n_splits)
    test_datasets = setup_dataset(data_path, split='test', batch_size=batch_size, n_splits=n_splits)

    metric = []
    for fold in range(n_splits):
        print(f"Training on fold {fold+1}/{n_splits}")
        print(f"Training on {len(train_datasets[fold][0].dataset)} samples")
        print(f"Testing on {len(test_datasets[fold][0].dataset)} samples")

        # Training
        model = setup_model()
        train_STL_model(model, train_datasets[fold], nb_epoch, Use_tb=True)

        # Testing
        metric_result = test_STL_model(model, test_datasets[fold], Use_tb=True)

        metric.append(metric_result)

    print(metric)
    
    return np.mean(metric, axis=0), np.std(metric, axis=0)


if __name__ == '__main__':
    n_splits = 5  # Number of folds for cross-validation
    nb_epoch = 100  # Number of training epochs
    batch_size = 1024

    stl.model_name_prefix = 'manual_CV_wi_feedback'
    stl.do_feedback = True
    data_path = 'data/STL/stl10_binary/'

    results = k_fold_cross_validation(n_splits, nb_epoch, batch_size, data_path)
    print('Mean: {0} \n Std: {1}'.format(*results))
    print(stl.model_name_prefix)
