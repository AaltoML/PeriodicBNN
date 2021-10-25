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
from laplace import KFACLaplace

from scipy.stats import norm
inv_probit = norm.cdf

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
  
##############################################################################################      
    
# Training for 10 splits in 10-fold cross-validation
def train_splits(train_sets, test_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args, save=False):

    if num_classes == 1:
        regression = True
    else:
        regression = False
    res = 0
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
        
        optimizer= torch.optim.SGD(param_list, lr=args.lr_adam, momentum=args.momentum)
        if num_classes == 1:
            columns = ["ep", "lr", "tr_loss", "tr_rmse", "te_loss", "te_rmse", "meas var", "lengthscale"]    
        else:
            columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "meas var", "lengthscale"]
        
        start_epoch = 0

        schedule = utils.schedule_step_linear
        
        for epoch in range(start_epoch, args.epochs):
            time_ep = time.time()
            
            #Select learning rate for the epoch
            lr_new, lr_adam_factor = schedule(epoch, split, args)
            utils.adjust_learning_rate(optimizer, lr_adam_factor, factor = True)

            #Train for one epoch
            train_res = utils.train_epoch(loaders["train"], model, loss_function.loss, optimizer, cuda=use_cuda, regression = regression)
            lr = args.lr_adam * lr_adam_factor

            #Evaluate on the test set
            if (
                (epoch == 0
                or epoch % args.eval_freq == args.eval_freq - 1
                or epoch == args.epochs - 1)
            ):
                test_res = utils.eval(loaders["test"], model, loss_function.loss, cuda=use_cuda, regression=regression)
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
            
        val_res = utils.eval(loaders["test"], model, loss_function.loss, cuda=use_cuda, regression=regression)
        res += val_res["accuracy"]

        if save:
            #Save checkpoint
            utils.save_checkpoint(
                args.dir,
                args.epochs,
                name="checkpoint_{}".format(split),
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            
    return res/len(train_sets)
    
def KFACLaplace_test(N, KFACmodel, loss_function, t_input, t_target, loader, scale, num_classes, use_variance, use_cuda):

    predictions = np.zeros((len(loader.dataset), num_classes))
    Fi = np.zeros((len(loader.dataset), num_classes))
    Fmu = np.zeros((len(loader.dataset), num_classes))
    Fmu_temp = np.zeros((len(loader.dataset), num_classes))
    Fvar = np.zeros((len(loader.dataset), num_classes))
    targets = np.zeros(len(loader.dataset))

    for i in range(N):

        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        KFACmodel.net.load_state_dict(KFACmodel.mean_state)

        if i == 0:
            KFACmodel.net.train()
            loss, _ = loss_function.loss(KFACmodel.net, t_input, t_target)
            loss.backward(create_graph=True)
            KFACmodel.step(update_params=False)

        KFACmodel.sample(scale=scale, cov=True)
        KFACmodel.eval()

        k = 0
        for input, target in loader:
            if use_cuda:
                input = input.cuda(non_blocking=True)

            torch.manual_seed(i)
            if use_cuda:
                torch.cuda.manual_seed(i)

            with torch.no_grad():
                output = KFACmodel.net(input)
                Fi[k : k + input.size()[0]] = output.cpu().numpy()
                if num_classes == 1:
                    predictions[k : k + input.size()[0]] += output.cpu().numpy()
                else:
                    predictions[k : k + input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
            targets[k : (k + target.size(0))] = target.numpy().squeeze()
            k += input.size()[0]
        Fmu_temp = Fmu + (Fi - Fmu) / (i + 1)
        Fvar = Fvar + (Fi - Fmu)*(Fi - Fmu_temp)
        Fmu = Fmu_temp

    Fvar = Fvar/(i+1)
    p = inv_probit(Fmu / np.sqrt(1 + Fvar))
    if use_variance:
        if num_classes == 1:
            predictions = Fmu
            predictive_variance = Fvar
            if KFACmodel.net.s is not None:
                s = KFACmodel.net.s.forward().detach().cpu().numpy()
            else:
                s = 0
        else:
            predictions = p
            predictive_variance = p - p**2
            s = 0
    else:
        predictions /= N
        predictive_variance = 0
        s = 0
        
    return predictions, predictive_variance, s, targets, Fvar

#######################################################################################
            
if __name__ == "__main__":

    class args:
        '''class for collecting some input parameters'''
        pass
    
    setup_ind = int(sys.argv[1]) #Setup index from command line arguments
    data_name = sys.argv[2] #Dataset name from command line arguments

    args.model = 'uci' # The model architecture to be used (options found in model.py)

    # Different model options to choose from using the 'setup_ind'
    dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
    kernels = ['ArcCosine','RBF','RBF','RBF','Matern','Matern','Matern','Matern','Matern','Matern','RBF','Matern','Matern','RBF','Matern','Matern']
    nus = [1, 1, 1, 1, 5/2, 5/2, 5/2, 3/2, 3/2, 3/2, 1 ,5/2, 3/2, 1, 5/2, 3/2]
    globalstats = [False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,True]
    periodics = ['sin','sin','sin','triangle','sin','sin','triangle','sin','sin','triangle','sincos','sincos','sincos','prelu','prelu','prelu']
        
    #### Define training setup
    args.dir = data_name+dir_names[setup_ind] # Directory for saving results
    args.dataset = data_name # The dataset to use: 'banana','diabetes', 'adult', 'connect-4', 'covtype', 'CIFAR10', 'CIFAR10_5class', 'MNIST', 'CIFAR_SVHN', 'CIFAR10_100'
    args.kernel = kernels[setup_ind] # GP covariance, either 'Matern', 'RBF' or 'ArcCosine'
    args.nu = nus[setup_ind] # Matern smoothness value
    args.global_stationary = globalstats[setup_ind]
    args.periodic_fun = periodics[setup_ind] #"triangle" or "sin"

    args.data_path = "../data/" # Path to dataset location
    args.epochs = 20 # Total number of epochs to run
    args.lr_init = 0# Not used for KFAC experiments, but needed as utils.py is shared with SWAG experiments

    args.milestones = True
    args.optimize_s_and_ell = True
    args.lengthscale = [1 for i in range(10)]

    args.momentum = 0.9 #SGD momentum (default: 0.9)
    args.meas_noise = False

    #### Usually fixed parameters
    args.seed = 1 # Random seed
    args.eval_freq = 5 # The epoch frequency for evaluating the model
    args.num_workers = 4 # Number of workers for Torch

    dropout_list = [0.1, 0.1, 0.1, 0.1] # Dropout rate for layers before the last hidden layer (individually for datasets)
    batch_size_list = [50, 500, 500, 500]
    reg_batch_size_list = [50, 50, 50, 500]
    reg_lengthscale_list = [5, 5, 5, 5]
    #'''

    np.random.seed(args.seed)

    args.device = None

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    datasets = ['diabetes', 'adult', 'connect-4', 'covtype']
    reg_datasets = ['boston', 'concrete', 'airfoil', 'elevators']
    args.lr_adam = None
    dropout = 0
    for di,dd in enumerate(datasets):
        if dd == args.dataset:
            args.lr_adam = 0.001
            args.ell_and_s_lr = 0.01
            dropout = dropout_list[di]
            args.batch_size = batch_size_list[di]
            args.epochs = 100
    for di,dd in enumerate(reg_datasets):
        if dd == args.dataset:
            args.lr_adam = 0.001
            args.ell_and_s_lr = 0.01
            dropout = dropout_list[di]
            args.batch_size = reg_batch_size_list[di]
            args.epochs = 100
            args.milestones = True
            args.optimize_s_and_ell = True
            if dir_names[setup_ind] == 'relu':
                args.lengthscale = [1 for i in range(10)]
            else:
                args.lengthscale = [reg_lengthscale_list[di] for i in range(10)]
            args.meas_noise = 1
    args.swa_start = args.epochs
    
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

    ################ Dataset loading ####################################
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    train_sets, test_sets, num_classes, D = dataset_maker.load_dataset(args.dataset, args.data_path, args.seed)

    N_train = len(train_sets[0])
    n_splits = len(train_sets)

    # use a slightly modified loss function that allows input of model
    if num_classes == 1:
        loss_function = L2_with_prior(N_train)
        use_variance = True
    else:
        loss_function = ce_with_prior(N_train)
        use_variance = False
        
    lr_grid = [0.00005, 0.0001, 0.0005, 0.001]
    scores = []
    small_train_sets = []
    val_sets = []
    for i in range(n_splits):
        s = 4*N_train//5
        small_train_sets.append(torch.utils.data.Subset(train_sets[i],range(0,s)))
        val_sets.append(torch.utils.data.Subset(train_sets[i],range(s,N_train)))
    for lr in lr_grid:
        args.lr_adam = lr
        with HiddenPrints():
            result = train_splits(small_train_sets, val_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args, save = False)
        if np.isnan(result):
            if num_classes == 1:
                result = 10000
            else:
                result = 0
        scores.append(result)
        print('lr of:',lr,'resulting in RMSE:',result)
        
    if num_classes == 1:
        best_lr = lr_grid[np.argmin(scores)]
    else:
        best_lr = lr_grid[np.argmax(scores)]
        
    args.lr_adam = best_lr
    print('best lr:', best_lr)
    result = train_splits(small_train_sets, val_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args, save = True)
    
    args.N = 50
    grid_N = 30

    scale_grid = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25] 
    eps = 1e-12
    scale_scores = np.zeros(len(scale_grid))
    use_ood_val = True
        
    if use_ood_val:
        ood_loader = dataset_maker.UniformNoise(args.dataset, delta=2, train=False, size=200, batch_size=args.batch_size)
    
    for split in range(len(train_sets)):

        args.file = args.dir + '/checkpoint_{}-100.pt'.format(split)

        train_set = small_train_sets[split]
        test_set = val_sets[split]
            
        trainloader = DataLoader(
                        train_set,
                        batch_size=args.batch_size,
                        shuffle=False,
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
    
        loaders = {
                    "train": trainloader,
                    "test": testloader,
                }    
    
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, dropout = dropout, kernel = args.kernel, nu = args.nu, lengthscale = args.lengthscale[split], device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, meas_noise = args.meas_noise, **model_cfg.kwargs)
        if use_cuda:
            model.cuda()

        print("Loading model %s" % args.file)
        checkpoint = torch.load(args.file, map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        KFACmodel = KFACLaplace(
            model, eps=5e-4, data_size=len(loaders["train"].dataset)
        )  # eps: weight_decay

        t_input, t_target = next(iter(loaders["train"]))

        if use_cuda:
            t_input, t_target = (
                t_input.cuda(non_blocking=True),
                t_target.cuda(non_blocking=True),
            )

        for s_i, scale in enumerate(scale_grid):
            predictions, predictive_variance, s, targets, Fvar = KFACLaplace_test(grid_N, KFACmodel, loss_function, t_input, t_target, loaders["test"], scale, num_classes, use_variance, use_cuda)
            
            if use_ood_val:
                preds_ood, predvar_ood, _, targets_ood, Fvar_ood = KFACLaplace_test(grid_N, KFACmodel, loss_function, t_input, t_target, ood_loader, scale, num_classes, use_variance, use_cuda)
                
            if num_classes == 1:
                log_lik_in = np.mean(norm.logpdf(predictions.squeeze() - targets.squeeze(), loc=0, scale=np.sqrt(s**2 + predictive_variance.squeeze())))
                if use_ood_val:
                    log_lik_out = np.mean(norm.logpdf(preds_ood.squeeze() - targets_ood.squeeze(), loc=0, scale=np.sqrt(s**2 + predvar_ood.squeeze())))
                else:
                    log_lik_out = 0
                log_lik = log_lik_in + log_lik_out/5
                print('log lik out:', log_lik_out, 'log_lik_in:', log_lik_in)
            else:
                y = targets.astype(np.compat.long)
                log_lik_in = -1*F.nll_loss(torch.from_numpy(np.log(predictions + 1e-8)), torch.from_numpy(y)).numpy()
                
                if use_ood_val:
                    log_lik_out = np.mean(np.log(preds_ood + 1e-8))
                else:
                    log_lik_out = 0
                log_lik = log_lik_in + log_lik_out/5
                print('log lik out:', log_lik_out, 'log_lik_in:', log_lik_in)
            scale_scores[s_i] += log_lik 
            
    print('scale scores:',scale_scores/n_splits)   
    best_scale = scale_grid[np.argmax(scale_scores)]
    
    print('best scale:',best_scale)
    
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    train_splits(train_sets, test_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args, save = True)
    
    for split in range(len(train_sets)):

        args.file = args.dir + '/checkpoint_{}-100.pt'.format(split)

        train_set = train_sets[split]
        test_set = test_sets[split]
            
        trainloader = DataLoader(
                        train_set,
                        batch_size=args.batch_size,
                        shuffle=False,
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
    
        loaders = {
                    "train": trainloader,
                    "test": testloader,
                }    
    
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, dropout = dropout, kernel = args.kernel, nu = args.nu, lengthscale = args.lengthscale[split], device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, meas_noise = args.meas_noise, **model_cfg.kwargs)
        if use_cuda:
            model.cuda()

        print("Loading model %s" % args.file)
        checkpoint = torch.load(args.file, map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])

        KFACmodel = KFACLaplace(
            model, eps=5e-4, data_size=len(loaders["train"].dataset)
        )  # eps: weight_decay

        t_input, t_target = next(iter(loaders["train"]))

        if use_cuda:
            t_input, t_target = (
                t_input.cuda(non_blocking=True),
                t_target.cuda(non_blocking=True),
            )

        predictions, predictive_variance, s, targets, Fvar = KFACLaplace_test(args.N, KFACmodel, loss_function, t_input, t_target, loaders["test"], best_scale, num_classes, use_variance, use_cuda)

        np.savez(args.dir+'/KFAC_result_{}.npz'.format(split), predictions=predictions, targets=targets, predictive_variance=predictive_variance, s=s, Fvar=Fvar)


        x = predictions.squeeze()
        if num_classes == 1:
            y = targets.squeeze()
        else:
            y = targets.squeeze().astype(np.compat.long)
        if num_classes == 1:
            var = predictive_variance.squeeze()
            nlpd = -1*np.mean(norm.logpdf(x - y, loc=0, scale=np.sqrt(s**2 + var)))
            print('NLPD:', nlpd)
            rmse = np.sqrt(np.mean((x - y)**2))
            print('RMSE:', rmse)
        else:
            nlpd = F.nll_loss(torch.from_numpy(np.log(predictions + 1e-8)), torch.from_numpy(y)).numpy()
            print('NLPD:', nlpd)
            acc = np.sum(np.argmax(x, axis = -1) == y)/y.shape[0]
            print('ACC:', acc)
