import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import numpy as np
import model as model_architecture
from losses import ce_with_prior
import utils
import dataset_maker
from torch.utils.data import DataLoader

##############################################################################################      
    
# Training for 10 splits in 10-fold cross-validation
def train_splits(train_sets, test_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args):

    if num_classes == 1:
        regression = True
    else:
        regression = False
    for split in range(len(train_sets)):

        train_set = train_sets[split]
        test_set = test_sets[split]

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

        print("Preparing model")
        
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, dropout = dropout, kernel = args.kernel, nu = args.nu, lengthscale = args.lengthscale[split], device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, meas_noise = args.meas_noise, **model_cfg.kwargs)
        model.to(args.device)
        
        param_list = []
        for param in model.parameters():
            if param.numel() == 1:
                if args.optimize_s_and_ell:
                    param_list.append({'params': param, 'lr': args.ell_and_s_lr})
                else:
                    pass
            elif param.requires_grad == True:
                param_list.append({'params': param})
        
        optimizer_adam = torch.optim.SGD(param_list, lr=args.lr_adam, momentum=args.momentum)

        columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "meas var", "lengthscale"]
        
        start_epoch = 0

        schedule = utils.schedule_step_gentle
        
        for epoch in range(start_epoch, args.epochs):
            time_ep = time.time()
            
            #Select learning rate for the epoch
            lr_new, lr_adam_factor = schedule(epoch, split, args)
            utils.adjust_learning_rate(optimizer_adam, lr_adam_factor, factor = True)

            #Train for one epoch
            train_res = utils.train_epoch(loaders["train"], model, loss_function.loss, optimizer_adam, cuda=use_cuda, regression = regression)
            lr = args.lr_adam * lr_adam_factor

            #Evaluate on the test set
            if (
                (epoch == 0
                or epoch % args.eval_freq == args.eval_freq - 1
                or epoch == args.epochs - 1)
            ):
                test_res = utils.eval(loaders["test"], model, loss_function.loss, cuda=use_cuda)
            else:
                test_res = {"loss": None, "accuracy": None}

            #Calculate epoch duration
            time_ep = time.time() - time_ep
            
            if use_cuda:
                memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
            else:
                memory_usage = 0
            
            try:
                s = model.s.forward().item()
            except:
                s = None
            try:
                ell = model.lengthscale.forward().item()
            except:
                ell = None
                
            #Print results for one epoch   
            values = [
                epoch + 1,
                lr,
                train_res["loss"],
                train_res["accuracy"],
                test_res["loss"],
                test_res["accuracy"],
                s,
                ell,
            ]
            table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.6f")
            if epoch % 40 == 0:
                table = table.split("\n")
                table = "\n".join([table[1]] + table)
            else:
                table = table.split("\n")[2]
            print(table)

        #Save checkpoint
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            state_dict=model.state_dict(),
            optimizer_adam=optimizer_adam.state_dict(),
        )
            
    return ell

#######################################################################################
            
if __name__ == "__main__":

    class args:
        '''class for collecting some input parameters'''
        pass
    
    setup_ind = int(sys.argv[1]) #Setup index from command line arguments
    data_name = 'MNIST'

    args.model = 'mnist' # The model architecture to be used (options found in model.py)

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

    BO_it = 5 # Number of Bayesian optimization iterations
    args.data_path = "../data/" # Path to dataset location
    args.epochs = 20 # Total number of epochs to run
    args.lr_init = 0# Not used for KFAC experiments, but needed as utils.py is shared with SWAG experiments
    args.batch_size = 50 #The batch size to be used
    args.milestones = True
    args.optimize_s_and_ell = True
    args.momentum = 0.9 #SGD momentum (default: 0.9)
    args.meas_noise = False

    #### Usually fixed parameters
    args.seed = 1 # Random seed
    args.eval_freq = 5 # The epoch frequency for evaluating the model
    args.num_workers = 4 # Number of workers for Torch

    np.random.seed(args.seed)

    args.device = None

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print("Preparing directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    model_cfg = getattr(model_architecture, args.model)

    if args.dataset == 'MNIST':
        args.batch_size = 64
        args.milestones = True
        args.lr_adam = 0.001
        args.ell_and_s_lr = 0.1*args.lr_adam
        args.epochs = 50
        dropout = 0.1
        args.lengthscale = [0.2]

    args.swa_start = args.epochs

    ################ Dataset loading ####################################
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    train_sets, test_sets, num_classes, D = dataset_maker.load_dataset(args.dataset, args.data_path, args.seed)

    N_train = len(train_sets[0])
    n_splits = len(train_sets)

    # use a slightly modified loss function that allows input of model
    loss_function = ce_with_prior(N_train)
 
    train_splits(train_sets, test_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args)
    
