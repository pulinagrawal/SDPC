import sys
from pathlib import Path

sys.path.append(str((Path("..").resolve().absolute())))

import numpy as np
#from LogGabor import LogGaborFit
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch
from SPC_2L.DataTools import DataBase
import pickle
from SPC_2L.DataTools import LCN, whitening, z_score, mask, to_cuda, norm
import torch.nn.functional as f
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import json

with open('../SPC_2L/param.json', 'r') as fp:
    config = json.load(fp)

def reconstruction(Net, images, Pursuit):
    nb_image = len(images)
    num_channels = 3 if config["Params"]["RBG"] else 1
    for i in range(Net.nb_layers - 1):
        if i == 0:
            output_size = (config["Params"]["Image_Size"][0] - config["Param"]["Dico_Shape"][0] + 2*config["Param"]["Out_Padding"][0])/config["Param"]["Stride"][0] + 1
        else:
            output_size = (output_size[i-1] - Net.layers[i].dico_shape[0] + 2*Net.layers[i].out_padding[0])/Net.layers[i].stride[0] + 1
    all_batch = torch.zeros(nb_image, num_channels, config["Params"]["Image_Size"][0], config["Params"]["Image_Size"][1]).cuda()
    save_gamma_0 = torch.zeros(nb_image,config["Network"]["Num_Features"][1],output_size[0], output_size[0]).cuda()
    save_gamma_1 = torch.zeros(nb_image,config["Network"]["Num_Features"][2],output_size[1],output_size[1]).cuda()
    k = 0
    for idx_batch, data in enumerate(images):
        batch = data[0].cuda()
        all_batch[k:k+batch.size(0),:,:,:] = batch
        gamma, it, Loss_G, delta = Pursuit.coding(batch)

        save_gamma_0[k:k+batch.size(0),:,:,:] = gamma[0][:,:,:,:]
        save_gamma_1[k:k+batch.size(0),:,:,:] = gamma[1][:,:,:,:]
        k+=batch.size(0)
        if k >= nb_image:
            break
    
    gamma_n = [save_gamma_0,save_gamma_1]
    
    reco = [None] * (Net.nb_layers)
    for i in range(Net.nb_layers-1,-1,-1):
        reco[i] = gamma_n[i]
        for j in range(i, -1, -1):
            reco[i] = Net.layers[j].backward(reco[i])
    reco = torch.stack(reco, dim=0)
    return reco.permute(1, 0, 2, 3)

def sparseRepresentation(Net, images, Pursuit):
    sparse_list = []
    for idx_batch, data in enumerate(images):
        batch = data[0].cuda()
        gamma, it, Loss_G, delta = Pursuit.coding(batch)

        layer_sparse_list = []
        for i in range(Net.nb_layers):
            sparse = (gamma[i]!=0).float()
            layer_sparse_list.append(sparse)

        sparse_tensor = torch.stack(layer_sparse_list, dim=0)
        sparse_list.append(sparse_tensor)

    return torch.stack(sparse_list, dim=0)