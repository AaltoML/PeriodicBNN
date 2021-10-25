import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import model as model_architecture
import dataset_maker
from torch.utils.data import DataLoader
import utils
import time
import random

from run_swag_experiment import ce_with_prior, L2_with_prior, train_splits

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from botorch.optim import optimize_acqf

class args:
    '''class for collecting some input parameters'''
    pass
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
setup_ind = int(sys.argv[1]) #Setup index from slurm array
data_name = 'CIFAR10'

# Different model options to choose from using the 'setup_ind'
dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
kernels = ['ArcCosine','RBF','RBF','RBF','Matern','Matern','Matern','Matern','Matern','Matern','RBF','Matern','Matern','RBF','Matern','Matern']
nus = [1, 1, 1, 1, 5/2, 5/2, 5/2, 3/2, 3/2, 3/2, 1 ,5/2, 3/2, 1, 5/2, 3/2]
globalstats = [False,False,True,True,False,True,True,False,True,True,True,True,True,True,True,True]
periodics = ['sin','sin','sin','triangle','sin','sin','triangle','sin','sin','triangle','sincos','sincos','sincos','prelu','prelu','prelu']
        
    
#### Frequently edited parameters
args.dir = data_name+dir_names[setup_ind] # Directory for saving results
args.model = "cifar" # The model architecture to be used (options found in model.py)
args.dataset = data_name # The dataset to use: 'banana','diabetes', 'adult', 'connect-4', 'covtype', 'CIFAR10', 'CIFAR10_5class', 'MNIST', 'CIFAR_SVHN', 'CIFAR10_100'
args.kernel = kernels[setup_ind] # GP covariance, either 'Matern', 'RBF' or 'ArcCosine'
args.nu = nus[setup_ind] # Matern smoothness value
args.global_stationary = globalstats[setup_ind]
args.periodic_fun = periodics[setup_ind] #"triangle" or "sin"

BO_it = 5 # Number of Bayesian optimization iterations
args.data_path = "../data/" # Path to dataset location
args.lr_init = 0.0001 # Training initial learning rate, unless Adam is used
args.swa_lr = [0.1] # Learning rate for SWA sampling phase
args.optimize_s_and_ell = True
args.lengthscale = [1]

#### Usually fixed parameters
args.seed = 1 # Random seed
args.eval_freq = 5 # The epoch frequency for evaluating the model
args.num_workers = 4 # Number of workers for Torch
args.max_num_models = 20 # The number of steps for which the model is averaged
args.resume = None # checkpoint to resume training from (default: None)
args.swa_c_epochs = 1 # SWA model collection frequency/cycle length in epochs (default: 1)
args.momentum = 0.9 #SGD momentum (default: 0.9)
args.five_fold_BO = False
args.grid_search = False


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

args.batch_size = 128
args.milestones = False
args.lr_adam = 0.0001
args.ell_and_s_lr = 0.1*args.lr_adam
args.MC_samples = 30
args.epochs = 90
args.swa_start = args.epochs - 40
dropout = 0.1

################ Dataset loading ####################################
print("Loading dataset %s from %s" % (args.dataset, args.data_path))
train_sets, test_sets, num_classes, D = dataset_maker.load_dataset(args.dataset, args.data_path, args.seed)

_, cifar10_test, _, _ = dataset_maker.load_dataset("CIFAR10", args.data_path, args.seed)
validation_set = cifar10_test
additional_val_set = True

N_train = len(train_sets[0])
n_splits = len(train_sets)

# use a slightly modified loss function that allows input of model
loss_function = ce_with_prior(N_train)

################ Bayesian optimization for learning rate ###############

#Shuffle index for train val split
shuffle_ind = list(range(0, N_train))
random.shuffle(shuffle_ind)
shuffle_ind = np.array(shuffle_ind)

def calculate_nlpd_score(swa_lr, train_data, val_data):
    nlpd_sum = 0
    args.swa_lr = [swa_lr]
    reps = 1
    for rep in range(reps):
        
        # Training for the swa_lr value
        with HiddenPrints():
            lengthscale = train_splits([train_data], [val_data], num_classes, D, model_cfg, use_cuda, loss_function, dropout, args)

        # Calculate NLL final result
        nlpd_i = utils.result_loader(args.dir, args.dataset, 1)

        nlpd_sum += nlpd_i
    score = nlpd_sum/reps
        
    return score, lengthscale
    
def run5_fold_score(swa_lr, train_data, five_fold = True):
    score_sum = 0
    if five_fold:
        folds = 5
    else:
        #Only single train validation split of 80/20 %
        folds = 1
    for i in range(folds):

        s = i*N_train//5
        e = (i+1)*N_train//5
            
        train_index = shuffle_ind[list(range(0, s))+list(range(e, N_train))]
        val_index = shuffle_ind[list(range(s, e))]
        
        train_set = torch.utils.data.Subset(train_data, train_index)
        val_set = torch.utils.data.Subset(train_data, val_index)
        
        score, lengthscale = calculate_nlpd_score(swa_lr, train_set, val_set)
        score_sum += score
    return -1*score_sum/folds, lengthscale
    
def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return model

def optimize_acqf_and_get_observation(acq_func, train_data, batch_size = 3):
    
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([
            torch.zeros(1), 
            torch.ones(1),
        ]), # A 2 x d tensor of lower and upper bounds for each column 
        q=batch_size, # The number of candidates.
        num_restarts=10, # The number of starting points for multistart acquisition 
        raw_samples=10, # The number of samples for initialization.
    )

    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    new_obj = torch.zeros(len(new_x),1)

    for i, lr in enumerate(new_x.squeeze()):
        if additional_val_set:
            val, _ = calculate_nlpd_score(lr, train_data, validation_set[0])
            new_obj[i] = -1*val
        else:
            new_obj[i], _ = run5_fold_score(lr, train_data, five_fold = args.five_fold_BO)

    return new_x, new_obj
    
time_start = time.time()

optimized_swa_lrs = []
optimized_lengthscales = []

# Do Bayesian optimization of swag learning rate for each split separately
for split in range(n_splits):

    print("START SPLIT {}".format(split+1))
    
    best_observed = []
    best_observed_lr = []

    # Bounds for SWA lr
    bounds = torch.tensor([[0.0001], [3]])

    # Initial grid search values for SWA learning rate (B model training data)
    if args.grid_search:
        lrs = torch.tensor([3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]).reshape(-1,1)
        BO_it = 0
    else:
        lrs = torch.tensor([3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001]).reshape(-1,1)

    values = torch.zeros(lrs.shape[0],1) # Initialize array for scores on initial grid
    
    training_data = train_sets[split]
    
    # Perform initial fit for the split
    args.resume = None
    old_ep = args.epochs
    args.epochs = args.swa_start
    if additional_val_set:
        _, ell = calculate_nlpd_score(0, training_data, validation_set[0])
    else:
        _, ell = run5_fold_score(0, training_data, five_fold = args.five_fold_BO)
    optimized_lengthscales.append(ell)
    args.epochs = old_ep
    args.resume = args.dir + "/checkpoint-{}.pt".format(args.swa_start)
        
    for i, lr in enumerate(lrs.squeeze()):
        if additional_val_set:
            val, _  = calculate_nlpd_score(lr, training_data, validation_set[0])
            values[i] = -1*val
        else:
            values[i], _ = run5_fold_score(lr, training_data, five_fold = args.five_fold_BO)
        print("tested SWAG learning rate", lr.item(), "resulting with NLPD", -1*values[i].item())
        
    values[torch.isnan(values)] = -1e6

    best_value = values.max().item()
    best_observed_x = lrs[values.argmax().item()].item()

    train_x = lrs
    train_obj = values

    best_observed_lr.append(best_observed_x)
    best_observed.append(best_value)

    print("Initial best LR {0} with mean NLPD {1}".format(best_observed_lr, -1*best_observed[0]))



    print("start Bayesian optimization")

    state_dict = None
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(BO_it):    

        print("Iteration number {}".format(iteration))
        
        # fit the model
        model = get_fitted_model(
            normalize(train_x, bounds=bounds), 
            standardize(train_obj), 
            state_dict=state_dict,
        )
        
        # define the qNEI acquisition module using a QMC sampler
        # Sampler for quasi-MC base samples using Sobol sequences.
        qmc_sampler = SobolQMCNormalSampler(num_samples=100)
        qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max())

        # optimize and get new observation
        new_x, new_obj = optimize_acqf_and_get_observation(qEI, training_data)

        # update training points
        train_x = torch.cat((train_x, new_x))
        new_obj[torch.isnan(new_obj)] = -1e6
        train_obj = torch.cat((train_obj, new_obj))

        # update progress
        best_value = train_obj.max().item()
        best_lr = train_x[train_obj.argmax().item()].item()

        best_observed.append(best_value)
        best_observed_lr.append(best_lr)
        
    print()
    print("Final best LR {0} with mean NLPD {1}".format(best_observed_lr[-1], -1*best_observed[-1]))
    print("Execution time: {}".format(time.time() - time_start))
      
    optimized_swa_lrs.append(best_observed_lr[-1])
        
print("SWAG learning rates:", optimized_swa_lrs)
print("lengthscales:", optimized_lengthscales)

args.resume = None
args.swa_lr = optimized_swa_lrs
args.lengthscale = optimized_lengthscales
args.optimize_s_and_ell = False
train_splits(train_sets, test_sets, num_classes, D, model_cfg, use_cuda, loss_function, dropout, args)

