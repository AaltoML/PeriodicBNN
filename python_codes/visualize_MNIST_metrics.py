# Code for visualizing the rotated MNIST experiments results. This requires that the predictions for models are calculated and saved.

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import argparse
import tikzplotlib
import sklearn
from sklearn.metrics import roc_auc_score

def NLPD_ood(x):
    N = x.shape[0]
    c = x.shape[1]
    x_max = np.max(x, axis = -1)
    nlpd = -1*np.mean(np.log((c/(c-1))*(1 - x_max)))
    return nlpd

dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'rbf_sincos', 'matern32_sincos']
fig_names = ['ReLU', 'local RBF', 'global RBF (sin)', 'global RBF (tri)', 'local Matern-5/2', 'global Matern-5/2 (sin)', 'global Matern-5/2 (tri)', 'local Matern-3/2', 'global Matern-3/2 (sin)', 'global Matern-3/2 (tri)','global RBF (sincos)', 'global Matern-3/2 (sincos)']

setup_ind_list = [0,1,2,10]

data_name = 'MNIST'
MAP_estimate = False

dirs = []
fig_labels = []

if MAP_estimate:
    map_name = '/MAPestimate'
else:
    map_name = ''
for setup_ind in setup_ind_list:
    dirs.append(data_name+dir_names[setup_ind]+map_name) # Directory for saving results
    fig_labels.append(fig_names[setup_ind])

colors = ['r','g','b','c','m']
angles = range(0,370,10)

for f,folder in enumerate(dirs):
    predlist = []
    labellist = []

    # Load standard test set results
    res = np.load('{}/KFAC_result_ood_0.npz'.format(folder))
    preds = res['predictions']
    labels = res['targets']
    predlist.append(preds)
    labellist.append(labels)

    # Load rotated results
    for i in range(36):
        res = np.load('{}/KFAC_result_grid_{}.npz'.format(folder, i))
        preds = res['predictions']
        labels = res['targets']
        predlist.append(preds)
        labellist.append(labels)

    accuracies = []
    mean_confidences = []
    nlpds = []
    std_confidences = []
    std_nlpds = []

    for i in range(len(predlist)):
        x = predlist[i]
        y = labellist[i].astype(np.long)
        confidences = np.max(x, axis=-1)
        mean_confidences.append(np.mean(confidences))
        std_confidences.append(np.std(confidences))
        pred_classes = np.argmax(x, axis=-1)
        hits = np.sum(y.astype(int) == pred_classes)
        accuracies.append(hits/y.shape[0])
        nlpds.append(F.nll_loss(torch.from_numpy(np.log(x)), torch.from_numpy(y)).numpy())
        std_nlpds.append(np.std(F.nll_loss(torch.from_numpy(np.log(x)), torch.from_numpy(y), reduction = 'none').numpy()))


    plt.figure(1)
    mean_confidences = np.array(mean_confidences)
    std_confidences = np.array(std_confidences)
    plt.plot(angles,mean_confidences, color = colors[f], label=fig_labels[f])
    plt.xlabel('rotation angle')
    plt.title('mean confidence')
    
    plt.figure(2)
    plt.plot(angles,accuracies, color = colors[f], label=fig_labels[f])
    plt.xlabel('rotation angle')
    plt.title('accuracy')
    
    plt.figure(3)
    nlpds = np.array(nlpds)
    std_nlpds = np.array(std_nlpds)
    plt.plot(angles,nlpds, color = colors[f], label=fig_labels[f])
    plt.xlabel('rotation angle')
    plt.title('NLPD')

plt.figure(1)
plt.legend()
plt.savefig('rotatedMNIST_confidence.png', format='png')

plt.figure(2)
plt.legend()
plt.savefig('rotatedMNIST_accuracy.png', format='png')

plt.figure(3)
plt.legend()
plt.savefig('rotatedMNIST_nlpd.png', format='png')

