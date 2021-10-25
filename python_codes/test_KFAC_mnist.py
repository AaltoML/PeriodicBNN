import os, sys
import tqdm
import gc

import torch
import torch.nn.functional as F
import numpy as np
import model as model_architecture
from losses import ce_with_prior, L2_with_prior
import utils
import dataset_maker
from torch.utils.data import DataLoader

from laplace import KFACLaplace

class args:
    '''class for collecting some input parameters'''
    pass

setup_ind = int(sys.argv[1]) #Setup index from command line arguments
test_type = sys.argv[2] #Whether using rotated set or normal set from command line arguments
if test_type == 'rotated':
    angle_ind = int(sys.argv[3]) #Index of rotation angle from command line arguments ()

data_name = 'MNIST' #Dataset name
args.model = 'mnist' # The model architecture to be used (options found in model.py: 'uci, 'mnist', 'banana', cifar')

# Different model options to choose from using the 'setup_ind'
dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
kernels = ['ArcCosine','RBF','RBF','RBF','Matern','Matern','Matern','Matern','Matern','Matern','RBF','Matern','Matern','RBF','Matern','Matern']
nus = [1, 1, 1, 1, 5/2, 5/2, 5/2, 3/2, 3/2, 3/2, 1 ,5/2, 3/2, 1, 5/2, 3/2]
globalstats = [False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,True]
periodics = ['sin','sin','sin','triangle','sin','sin','triangle','sin','sin','triangle','sincos','sincos','sincos','prelu','prelu','prelu']
    
#### Frequently edited parameters
args.dir = data_name+dir_names[setup_ind] # Directory for saving results
args.dataset = data_name # The dataset to use: 'banana','diabetes', 'adult', 'connect-4', 'covtype', 'CIFAR10', 'CIFAR10_5class', 'MNIST', 'CIFAR_SVHN', 'CIFAR10_100'
args.kernel = kernels[setup_ind] # GP covariance, either 'Matern', 'RBF' or 'ArcCosine'
args.nu = nus[setup_ind] # Matern smoothness value
args.global_stationary = globalstats[setup_ind]
args.periodic_fun = periodics[setup_ind] #"triangle" or "sin"

args.data_path = "../data/" # Path to dataset location

files = os.listdir(args.dir)
checkpoint_files = [k for k in files if 'checkpoint' in k]
args.file = args.dir + '/' + checkpoint_files[-1]


args.batch_size = 64
dropout = 0.1
args.lengthscale = [1 for i in range(37)]
args.num_workers = 4
use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

args.N = 30
args.scale = 1.0
args.seed = 1

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll

def NLPD_ood(x):
    N = x.shape[0]
    c = x.shape[1]
    x_max = np.max(x, axis = -1)
    nlpd = -1*np.mean(np.log((c/(c-1))*(1 - x_max)))
    return nlpd

eps = 1e-12

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(model_architecture, args.model)

################ Dataset loading ####################################
print("Loading dataset %s from %s" % (args.dataset, args.data_path))
train_sets, test_sets, num_classes, D = dataset_maker.load_dataset(args.dataset, args.data_path, args.seed)
_, test_grid, num_classes, D = dataset_maker.load_dataset('rotated_MNIST', args.data_path, args.seed)
    
N_train = len(train_sets[0])
num_splits = [len(test_sets), len(test_grid)]

# use a slightly modified loss function that allows input of model
if num_classes == 1:
    loss_function = L2_with_prior(N_train)
else:
    loss_function = ce_with_prior(N_train)
    

#OOM management indexes
if test_type == 'standard':
    ds = 0
    iter2 = range(num_splits[0])
elif test_type == 'rotated':
    ds = 1
    iter2 = [angle_ind]


for split in iter2:

    train_set = train_sets[0]
    if ds == 0:
        test_set = test_sets[split]
        name = 'ood'
    elif ds == 1:
        test_set = test_grid[split]
        name = 'grid'
        
    trainloader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
    testloader = DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
    
    ###############################################################################  
    loaders = {
                "train": trainloader,
                "test": testloader,
            }

    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, dropout = dropout, kernel = args.kernel, nu = args.nu, lengthscale = args.lengthscale[split], device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, **model_cfg.kwargs)
    if use_cuda:
        model.cuda()

    print("Loading model %s" % args.file)
    checkpoint = torch.load(args.file)
    model.load_state_dict(checkpoint["state_dict"])

    print(len(loaders["train"].dataset))
    KFACmodel = KFACLaplace(model, eps=5e-4, data_size=len(loaders["train"].dataset))

    t_input, t_target = next(iter(loaders["train"]))
    if use_cuda:
        t_input, t_target = (
            t_input.cuda(non_blocking=True),
            t_target.cuda(non_blocking=True),
        )

    predictions = np.zeros((len(loaders["test"].dataset), num_classes))
    targets = np.zeros(len(loaders["test"].dataset))
    print(targets.size)

    for i in range(args.N):
        print("%d/%d" % (i + 1, args.N))
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        KFACmodel.net.load_state_dict(KFACmodel.mean_state)

        if i == 0:
            KFACmodel.net.train()
            print('calculating loss')
            loss, _ = loss_function.loss(KFACmodel.net, t_input, t_target)
            loss.backward(create_graph=True)
            print('START STEP')
            KFACmodel.step(update_params=False)

        KFACmodel.sample(scale=args.scale, cov=True)
        KFACmodel.eval()

        k = 0
        for input, target in tqdm.tqdm(loaders["test"]):
            if use_cuda:
                input = input.cuda(non_blocking=True)

            with torch.no_grad():
                output = KFACmodel.net(input)
                predictions[k : k + input.size()[0]] += (
                    F.softmax(output, dim=1).cpu().numpy()
                )
            targets[k : (k + target.size(0))] = target.numpy()
            k += input.size()[0]

        #nll is sum over entire dataset
        print("NLL:", NLPD_ood(predictions / (i + 1),))
    predictions /= args.N

    entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
    np.savez(args.dir+'/KFAC_result_{}_{}.npz'.format(name,split), entropies=entropies, predictions=predictions, targets=targets)

