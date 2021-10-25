#This file calculates AUC and AUPR values for the CIFAR-10 OOD detection experiment. This requires that the model testing script has already been run and the results are saved.

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.lines as mlines
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import tikzplotlib
import dataset_maker
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
    
dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']

setup_ind = int(sys.argv[1]) #Setup index from slurm array
folder = dir_names[setup_ind]

# Load OOD results
results_ood1 = np.load('CIFAR10{}/swag_ens_preds_CIFAR10_100_0.npz'.format(folder))
var_ood1 = results_ood1['pred_variance']

var1 = np.mean(var_ood1, axis = 1)


results_ood2 = np.load('CIFAR10{}/swag_ens_preds_CIFAR_SVHN_0.npz'.format(folder))
var_ood2 = results_ood2['pred_variance']

var2 = np.mean(var_ood2, axis = 1)


# Load in-distribution results
results_id = np.load('CIFAR10{}/swag_ens_preds_CIFAR10_0.npz'.format(folder))
var_id = results_id['pred_variance']

var_id = np.mean(var_id, axis = 1)


L1 = len(var1)
L2 = len(var2)
L_id = len(var_id)

labels_cifar10_svhn = np.concatenate((np.zeros(L_id), np.ones(L2)))
labels_cifar10_cifar100 = np.concatenate((np.zeros(L_id), np.ones(L1)))
variances_cifar10_svhn = np.concatenate((var_id, var2))
variances_cifar10_cifar100 = np.concatenate((var_id, var1))


print(folder)

## AUC

AUROC_10vs100_var = roc_auc_score(labels_cifar10_cifar100, variances_cifar10_cifar100)
print('OOD AUROC cifar10 vs cifar100, from variance:', "%.3f" % AUROC_10vs100_var)

AUROC_10vsSVHN_var = roc_auc_score(labels_cifar10_svhn, variances_cifar10_svhn)
print('OOD AUROC cifar10 vs SVHN, from variance:', "%.3f" % AUROC_10vsSVHN_var)


## AUPR

precision, recall, _ = precision_recall_curve(labels_cifar10_cifar100, variances_cifar10_cifar100)
aupr_100 = auc(recall, precision)
print('OOD AUPR cifar10 vs cifar100, from variance:', "%.3f" % aupr_100)

precision, recall, _ = precision_recall_curve(labels_cifar10_svhn, variances_cifar10_svhn)
aupr_svhn = auc(recall, precision)
print('OOD AUPR cifar10 vs SVHN, from variance:', "%.3f" % aupr_svhn)

