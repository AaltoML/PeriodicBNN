# This code is obtained from https://github.com/wjmaddox/swa_gaussian with major modifications. This GitHub repository is based on the paper: 'A Simple Baseline for Bayesian Uncertainty in Deep Learning' by Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson

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
        
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, D=D, dropout = dropout, kernel = args.kernel, nu = args.nu, lengthscale = args.lengthscale[split], device = args.device, global_stationary = args.global_stationary, periodic_fun = args.periodic_fun, **model_cfg.kwargs)
        model.to(args.device)

        print("SWAG training")
        swag_model = SWAG(
            model_cfg.base,
            no_cov_mat=False,
            max_num_models=args.max_num_models,
            *model_cfg.args,
            num_classes=num_classes,
            D=D,
            dropout=dropout,
            kernel = args.kernel,
            nu = args.nu,
            device = args.device,
            lengthscale = args.lengthscale[split],
            global_stationary = args.global_stationary,
            periodic_fun = args.periodic_fun,
            **model_cfg.kwargs
        )
        swag_model.to(args.device)

        
        last_layer_parameters = []
        last_layer_parameters.extend(model.fc_h.parameters())
        last_layer_parameters.extend(model.fc_o.parameters())
        
        param_list = []
        for param in model.parameters():
            if param.numel() == 1:
                if args.optimize_s_and_ell:
                    param_list.append({'params': param, 'lr': args.ell_and_s_lr})
                else:
                    pass
            elif param.requires_grad == True:
                param_list.append({'params': param})
                
        optimizer = torch.optim.SGD(last_layer_parameters, lr=args.lr_init, momentum=args.momentum)
        
        #optimizer_adam = torch.optim.Adam(param_list, lr=args.lr_adam)
        optimizer_adam = torch.optim.SGD(param_list, lr=args.lr_adam, momentum=args.momentum)

        columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "meas var", "lengthscale"]#"time", "mem_usage"]
        
        # Load previous model if continuing from checkpoint
        start_epoch = 0
        if args.resume is not None:
            print("Resume training from %s" % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_adam.load_state_dict(checkpoint["optimizer_adam"])

        schedule = utils.schedule_step
        
        for epoch in range(start_epoch, args.epochs):
            time_ep = time.time()
            
            #Select learning rate for the epoch
            lr_new, lr_adam_factor = schedule(epoch, split, args)
            utils.adjust_learning_rate(optimizer, lr_new)
            utils.adjust_learning_rate(optimizer_adam, lr_adam_factor, factor = True)


            #Train for one epoch
            if (epoch + 1) > args.swa_start:
                train_res = utils.train_epoch(loaders["train"], model, loss_function.loss, optimizer, cuda=use_cuda)
                lr = lr_new
            else:
                train_res = utils.train_epoch(loaders["train"], model, loss_function.loss, optimizer_adam, cuda=use_cuda)
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

            #Collect SWAG model
            if (
                (epoch + 1) > args.swa_start
                and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
            ):
                print("collecting SWAG")
                swag_model.collect_model(model)

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
            optimizer=optimizer.state_dict(),
            optimizer_adam=optimizer_adam.state_dict(),
        )
        if args.epochs > args.swa_start:
            utils.save_checkpoint(
                args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
            )

        if s is not None:
            mean_list1 , _ , _ = swag_model.generate_mean_var_covar()
            meas_var = mean_list1[-1].cpu()
        else:
            meas_var = 0
        
        swag_ens_preds = None
        swag_targets = None
        n_ensembled = 0.0
        if (epoch + 1) > args.swa_start:
            for epoch in range(0, args.MC_samples):
                    
                swag_model.sample(scale=1.0, cov=True, block=False, fullrank=True)
                utils.bn_update(loaders["train"], swag_model, cuda=use_cuda)
                swag_res = utils.predict(loaders["test"], swag_model, cuda = use_cuda, regression = regression)
                        
                swag_preds = swag_res["predictions"]
                swag_targets = swag_res["targets"]
                print("updating swag_ens",epoch)
                if swag_ens_preds is None:
                    swag_ens_preds = swag_preds.copy()
                    swag_ens_var = np.zeros_like(swag_ens_preds)
                else:
                    # rewrite in a numerically stable way
                    swag_ens_preds_new = swag_ens_preds + (swag_preds - swag_ens_preds) / (n_ensembled + 1)
                    swag_ens_var = swag_ens_var + (swag_preds - swag_ens_preds)*(swag_preds - swag_ens_preds_new)
                    swag_ens_preds = swag_ens_preds_new
                n_ensembled += 1
                
            #Save results
            np.savez(
                os.path.join(args.dir, "swag_ens_preds_{}_{}.npz".format(args.dataset, split)),
                predictions=swag_ens_preds,
                targets=swag_targets,
                pred_variance = swag_ens_var/n_ensembled,
                meas_variance = meas_var,
            )
            
    return ell
    
