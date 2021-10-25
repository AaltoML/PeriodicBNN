# This file creates predictive entropy and predictive marginal variance histograms for the CIFAR-10 OOD detection experiment. This file also picks and saves example images of most/least similar samples based on predictive entropy values. This requires that the model testing script has already been run and the results are saved.

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

def imshow(img, fig, ax, title):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(title)

#Entropy function
def entropy(x):
    Ex = -1*np.sum(x*np.log(x + 10**-8), axis=-1)
    return Ex
    
def NLPD_ood(x):
    N = x.shape[0]
    c = x.shape[1]
    x_max = np.max(x, axis = -1)
    nlpd = -1*np.mean(np.log((c/(c-1))*(1 - x_max)))
    return nlpd

def plot_histograms(x1, x2, x3, fig, ax, mode):
    log_y = False
    
    tot_min = min(np.min(x1), np.min(x2), np.min(x2))
    tot_max = max(np.max(x1), np.max(x2), np.max(x2))
    #Linear bins
    bins=np.linspace(tot_min, tot_max, 50)

    (counts1, bins) = np.histogram(x1, bins=bins)
    (counts2, bins) = np.histogram(x2, bins=bins)
    (counts3, bins) = np.histogram(x3, bins=bins)

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    mean3 = np.mean(x3)

    counts1 = counts1/len(x1)
    counts2 = counts2/len(x2)
    counts3 = counts3/len(x3)

    filter1 = np.ones(5)/5
    conv1 = np.convolve(counts1, filter1, 'same')
    conv2 = np.convolve(counts2, filter1, 'same')
    conv3 = np.convolve(counts3, filter1, 'same')

    for i in range(4):
        conv1[i] = counts1[i]
        conv2[i] = counts2[i]
        conv3[i] = counts3[i]
        conv1[-i-1] = counts1[-i-1]
        conv2[-i-1] = counts2[-i-1]
        conv3[-i-1] = counts3[-i-1]

    x = (bins + (bins[1] - bins[0])/2)[:-1]
    idx1 = len(x[x < mean1])-1
    idx2 = len(x[x < mean2])-1
    idx3 = len(x[x < mean3])-1

    y1 = conv1[idx1] + (mean1 - x[idx1])*(conv1[idx1+1] - conv1[idx1])/(x[idx1+1] - x[idx1])
    y2 = conv2[idx2] + (mean2 - x[idx2])*(conv2[idx2+1] - conv2[idx2])/(x[idx2+1] - x[idx2])
    y3 = conv3[idx3] + (mean3 - x[idx3])*(conv3[idx3+1] - conv3[idx3])/(x[idx3+1] - x[idx3])
        
    ax.hist(bins[:-1], bins, weights=counts1, color = 'r', alpha = 0.2)
    ax.hist(bins[:-1], bins, weights=counts2, color = 'y', alpha = 0.2)
    ax.hist(bins[:-1], bins, weights=counts3, color = 'g', alpha = 0.2)

    ax.plot(x,conv1,'r', label='SVHN')
    ax.plot(x,conv2,'y', label='CIFAR100')
    ax.plot(x,conv3,'g', label='CIFAR10')
    
    ax.stem([mean1], [y1], markerfmt=' ', linefmt='r', use_line_collection=True)
    ax.stem([mean2], [y2], markerfmt=' ', linefmt='y', use_line_collection=True)
    ax.stem([mean3], [y3], markerfmt=' ', linefmt='g', use_line_collection=True)
    
    if mode == 'ent':
        ax.set_xlim(0, 2.3)
    else:
        ax.set_xlim(0, 0.025)
    ax.set_ylim(0, 0.2)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']

setup_ind = int(sys.argv[1]) #Setup index from slurm array
folder = dir_names[setup_ind]

classnames = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_path = "../data/"
seed = 1

_, cifar10, _, _ = dataset_maker.load_dataset("CIFAR10", data_path, seed)
cifar10 = cifar10[0]
    
_, cifar100, _, _ = dataset_maker.load_dataset("CIFAR10_100", data_path, seed)
cifar100 = cifar100[0]
        
_, svhn, _, _ = dataset_maker.load_dataset("CIFAR_SVHN", data_path, seed)
svhn = svhn[0]


# Load OOD results
results_ood1 = np.load('CIFAR10{}/swag_ens_preds_CIFAR10_100_0.npz'.format(folder))
preds_ood1 = results_ood1['predictions']
labels_ood1 = results_ood1['targets']
var_ood1 = results_ood1['pred_variance']

x_ood1 = preds_ood1
y_ood1 = labels_ood1
var1 = np.mean(var_ood1, axis = 1)

nlpd_ood1 = NLPD_ood(x_ood1)

results_ood2 = np.load('CIFAR10{}/swag_ens_preds_CIFAR_SVHN_0.npz'.format(folder))
preds_ood2 = results_ood2['predictions']
labels_ood2 = results_ood2['targets']
var_ood2 = results_ood2['pred_variance']

x_ood2 = preds_ood2
y_ood2 = labels_ood2
var2 = np.mean(var_ood2, axis = 1)

nlpd_ood2 = NLPD_ood(x_ood2)

# Load in-distribution results
results_id = np.load('CIFAR10{}/swag_ens_preds_CIFAR10_0.npz'.format(folder))
preds_id = results_id['predictions']
labels_id = results_id['targets']
var_id = results_id['pred_variance']

x_id = preds_id
y_id = labels_id
var_id = np.mean(var_id, axis = 1)
print(folder)
class_accs = []
for i in range(10):
    ind = y_id == i
    xi = x_id[ind]
    yi = y_id[ind]
    preds_i = np.argmax(xi, axis=-1)
    hits_i = np.sum(yi.astype(int) == preds_i)
    acc_i = hits_i/yi.shape[0]
    print("class {} accuracy: {}".format(i, acc_i))


preds = np.argmax(x_id, axis=-1)
hits = np.sum(y_id.astype(int) == preds)
accuracy_id = hits/y_id.shape[0]


print('ACC id: ',accuracy_id)
print('')

ent1 = entropy(x_ood1)
ent2 = entropy(x_ood2)
ent_id = entropy(x_id)

############### Plotting #####################################

ind1 = np.argsort(ent1)
ind2 = np.argsort(ent2)
ind_id = np.argsort(ent_id)

N = 10

cifar10_high = ind_id[-N:]
cifar10_low = ind_id[0:N]
cifar100_high = ind1[-N:]
cifar100_low = ind1[0:N]
svhn_high = ind2[-N:]
svhn_low = ind2[0:N]

print(len(cifar10_high))
print(len(cifar10))
print(len(ind1))

os.makedirs('./cifarimgs/{}'.format(folder), exist_ok=True)
# Save images
for i in range(N):
    im_cifar10_high, _ = cifar10[int(cifar10_high[i])]
    im_cifar10_low, _ = cifar10[int(cifar10_low[i])]
    im_cifar100_high, _ = cifar100[int(cifar100_high[i])]
    im_cifar100_low, _ = cifar100[int(cifar100_low[i])]
    im_svhn_high, _ = svhn[int(svhn_high[i])]
    im_svhn_low, _ = svhn[int(svhn_low[i])]

    #Save CIFAR10 images
    image_k1 = im_cifar10_high.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/cifar10_image_high{}.png'.format(folder,i))
    
    image_k1 = im_cifar10_low.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/cifar10_image_low{}.png'.format(folder,i))

    #Save CIFAR100 images
    image_k1 = im_cifar100_high.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/cifar100_image_high{}.png'.format(folder,i))
    
    image_k1 = im_cifar100_low.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/cifar100_image_low{}.png'.format(folder,i))
    
    #Save SVHN images
    image_k1 = im_svhn_high.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/svhn_image_high{}.png'.format(folder,i))
    
    image_k1 = im_svhn_low.numpy().transpose(1,2,0)
    rescaled1 = (255.0 / image_k1.max() * (image_k1 - image_k1.min())).astype(np.uint8)
    im1 = Image.fromarray(rescaled1)
    im1.save('./cifarimgs/{}/svhn_image_low{}.png'.format(folder,i))


############################### Histograms ##########################

fig, ax = plt.subplots()
plot_histograms(ent2, ent1, ent_id, fig, ax, 'ent')
plt.legend()

filename = 'cifarimgs/'+folder+'_entropy'
plt.savefig(filename+'.png')

fig, ax = plt.subplots()
plot_histograms(var2, var1, var_id, fig, ax, 'var')
plt.legend()

filename = 'cifarimgs/'+folder+'_variance'
plt.savefig(filename+'.png')
