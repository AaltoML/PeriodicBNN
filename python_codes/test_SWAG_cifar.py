import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import numpy as np
import model as model_architecture
from losses import ce_with_prior, L2_with_prior
import utils
import dataset_maker
from torch.utils.data import DataLoader

from swag_posterior import SWAG        

##############################################################################################      
    
# Training for 10 splits in 10-fold cross-validation
def predict_splits(test_sets, num_classes, D, model_cfg, use_cuda, args):

    if num_classes == 1:
        regression = True
    else:
        regression = False
    for split in range(len(test_sets)):

        test_set = test_sets[split]

        testloader = DataLoader(
                        test_set,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

        ###############################################################################  
        loaders = {
                    "test": testloader,
                }

        print("Preparing model")
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, kernel = args.kernel, nu = args.nu, device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, **model_cfg.kwargs)
        model.to(args.device)

        # Load previous model from checkpoint
        print("Load collected model from %s" % args.resume)
        checkpoint = torch.load(args.resume)

        model.load_state_dict(checkpoint["state_dict"])

        # Initialize the SWAG model
        checkpoint_swag = torch.load(args.swa_resume)
        swag_model = SWAG(
            model_cfg.base,
            no_cov_mat=False,
            max_num_models=args.max_num_models,
            *model_cfg.args,
            num_classes=num_classes,
            D=D,
            kernel = args.kernel,
            nu = args.nu,
            device = args.device,
            global_stationary = args.global_stationary,
            periodic_fun = args.periodic_fun,
            **model_cfg.kwargs
        )
        swag_model.to(args.device)
        swag_model.load_state_dict(checkpoint_swag["state_dict"])

        swag_ens_preds = None
        swag_targets = None
        n_ensembled = 0.0

        for epoch in range(0, args.MC_samples):
            time_ep = time.time()
                
            swag_model.sample(scale=1.0, cov=False, block=False, fullrank=True)
            utils.bn_update(loaders["test"], swag_model, cuda=use_cuda)
            swag_res = utils.predict(loaders["test"], swag_model, cuda = use_cuda, regression = regression)
                    
            swag_preds = swag_res["predictions"]
            swag_targets = swag_res["targets"]

            if swag_ens_preds is None:
                swag_ens_preds = swag_preds.copy()
                swag_ens_var = np.zeros_like(swag_ens_preds)
            else:
                # rewrite in a numerically stable way
                swag_ens_preds_new = swag_ens_preds + (swag_preds - swag_ens_preds) / (n_ensembled + 1)
                swag_ens_var = swag_ens_var + (swag_preds - swag_ens_preds)*(swag_preds - swag_ens_preds_new)
                swag_ens_preds = swag_ens_preds_new
            n_ensembled += 1

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
            print('Epoch {} completed.'.format(epoch))


        if s is not None:
            mean_list1 , _ , _ = swag_model.generate_mean_var_covar()
            meas_var = mean_list1[-1].cpu()
        else:
            meas_var = 0

        #Save results
        np.savez(
            os.path.join(args.dir, "swag_ens_preds_{}_{}.npz".format(args.dataset, split)),
            predictions=swag_ens_preds,
            targets=swag_targets,
            pred_variance = swag_ens_var/n_ensembled,
            meas_variance = meas_var,
        )

#######################################################################################
            
if __name__ == "__main__":

    class args:
        '''class for collecting some input parameters'''
        pass

    # Different model options to choose from using the 'setup_ind'
    dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
    kernels = ['ArcCosine','RBF','RBF','RBF','Matern','Matern','Matern','Matern','Matern','Matern','RBF','Matern','Matern','RBF','Matern','Matern']
    nus = [1, 1, 1, 1, 5/2, 5/2, 5/2, 3/2, 3/2, 3/2, 1 ,5/2, 3/2, 1, 5/2, 3/2]
    globalstats = [False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,True]
    periodics = ['sin','sin','sin','triangle','sin','sin','triangle','sin','sin','triangle','sincos','sincos','sincos','prelu','prelu','prelu']
        
    setup_ind = int(sys.argv[1]) #Setup index from command line arguments
    data_name = sys.argv[2] #Dataset name from command line arguments
    
    #### Frequently edited parameters
    args.dir = 'CIFAR10'+dir_names[setup_ind] # Directory for saving results
    args.model = 'cifar' # The model architecture to be used (options found in model.py: 'uci, 'mnist', 'banana', cifar')
    args.dataset = data_name # The dataset to use
    args.kernel = kernels[setup_ind] # GP covariance, either 'Matern', 'RBF' or 'ArcCosine'
    args.nu = nus[setup_ind] # Matern smoothness value
    args.global_stationary = globalstats[setup_ind]
    args.periodic_fun = periodics[setup_ind] #"triangle" or "sin"

    args.data_path = "./data/" # Path to dataset location

    args.MC_samples = 30 #Number of samples for model averaging
    args.batch_size = 500
    args.max_num_models = 20 # The number of steps for which the model is averaged
    
    #### Usually fixed parameters
    args.seed = 1 # Random seed
    args.num_workers = 4 # Number of workers for Torch

    np.random.seed(args.seed)

    args.device = None

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    model_cfg = getattr(model_architecture, args.model)

    files = os.listdir(args.dir)
    swag_files = [k for k in files if 'swag-' in k]
    args.swa_resume = args.dir + '/' + swag_files[-1]
    args.resume = None
    print(args.resume)

    ################ Dataset loading ####################################
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    _, test_set, num_classes, D = dataset_maker.load_dataset(args.dataset, args.data_path, args.seed)

    predict_splits(test_set, num_classes, D, model_cfg, use_cuda, args)
    print("Finished setup number {}".format(setup_ind))

