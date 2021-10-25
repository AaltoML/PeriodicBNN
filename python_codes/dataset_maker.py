import pandas as pd
import pickle
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#########################################################################

class banana_dataset(Dataset):
    def __init__(self, x, y, labels):
        super(banana_dataset).__init__()
        self.x = x
        self.y = y
        self.labels = labels
    def __getitem__(self, index):
        point = torch.FloatTensor((self.x[index],self.y[index]))
        label = int(self.labels[index])
        return point, label
    def __len__(self):
        return len(self.labels)
        
class UCI_dataset(Dataset):
    def __init__(self, x, labels):
        super(UCI_dataset).__init__()
        self.x = x
        self.labels = labels
    def __getitem__(self, index):
        features = self.x[index,:]
        label = self.labels[index]
        return features, label
    def __len__(self):
        return self.labels.shape[0]
        
# UCI dataset loading function
def load_UCI_dataset(full_path): 
    dataframe = pd.read_csv(full_path, header=None, na_values='?') # load the dataset as a numpy array
    dataframe = dataframe.dropna() # drop rows with missing values
    last_ix = len(dataframe.columns) - 1 
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix] # split into inputs and outputs
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix
    
# UCI regression dataset loading function
def load_UCIreg_dataset(full_path, name):
    # load the dataset as a numpy array
    if name == 'boston':
        df = pd.read_csv(r""+full_path+'.csv')
        Xo = df[['crim','zn','indus','chas','nox','rm','age','dis','tax','ptratio','black','lstat']].to_numpy()
        Yo = df['medv'].to_numpy().reshape((-1,1))
    elif name == 'concrete':
        df = pd.read_csv(r""+full_path+'.csv')
        Xo = df[['cement','water','coarse_agg','fine_agg','age']].to_numpy()
        Yo = df['compressive_strength'].to_numpy().reshape((-1,1))
    elif name == 'airfoil':
        df = pd.read_csv(r""+full_path+'.csv')
        Xo = df[['Frequency','AngleAttack','ChordLength','FreeStreamVelocity','SuctionSide']].to_numpy()
        Yo = df['Sound'].to_numpy().reshape((-1,1))
    elif name == 'elevators':
        # Load all the data

        data = np.array(loadmat(full_path+'.mat')['data'])
        Xo = data[:, :-1]
        Yo = data[:, -1].reshape(-1,1)
    return Xo, Yo

def load_dataset(name, datapath, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Training data
    if name == 'banana':
        data = pd.read_csv(datapath+"banana_datapoints.csv", header = None, prefix = 'col') 
        classes = pd.read_csv(datapath+"banana_classes.csv", header = None, prefix = 'col')
        n = len(classes.col0)
        train_set = [banana_dataset([data.col0[i] for i in range(n)], [data.col1[i] for i in range(n)], [classes.col0[i] for i in range(n)])]

        #Test data
        gridwidth = 200
        gridlength = 3.75
        x_vals = np.linspace(-gridlength,gridlength,gridwidth)
        y_vals = np.linspace(-gridlength,gridlength,gridwidth)
        grid_samples = np.zeros((gridwidth*gridwidth,2))
        for i in range(gridwidth):
            for j in range(gridwidth):
                grid_samples[i*gridwidth + j, 0] = x_vals[i]
                grid_samples[i*gridwidth + j, 1] = y_vals[j]

        grid_set = torch.from_numpy(grid_samples).float()
        big_n = grid_set.shape[0]

        test_set = [banana_dataset([grid_set[i,0] for i in range(big_n)], [grid_set[i,1] for i in range(big_n)], [0 for i in range(big_n)])]
        
        num_classes = 2
        D = 2
        
    elif name == "CIFAR10":
        
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()]) 
        
        transform_validation = transforms.Compose([transforms.ToTensor()])
        
        trainset = torchvision.datasets.CIFAR10(root=datapath+"CIFAR10", train=True, transform=transform_train, download=True)
        testset = torchvision.datasets.CIFAR10(root=datapath+"CIFAR10", train=False, transform=transform_validation, download=True)
        
        train_set = [trainset]
        test_set  = [testset]
        num_classes = 10
        D = 32*32

    elif name == "CIFAR_SVHN":
        
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()]) 
        
        transform_validation = transforms.Compose([transforms.ToTensor()])
        
        trainset = torchvision.datasets.CIFAR10(root=datapath+"CIFAR10", train=True, transform=transform_train, download=True)
        testset = torchvision.datasets.SVHN(root=datapath+"SVHN", split='test', transform=transform_validation, download=True)
        
        train_set = [trainset]
        test_set  = [testset]
        num_classes = 10
        D = 32*32
        
    elif name == "CIFAR10_100":
        
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        
        transform_validation = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root=datapath+"CIFAR10", train=True, transform=transform_train, download=True)
        testset = torchvision.datasets.CIFAR100(root=datapath+"CIFAR100", train=False, transform=transform_validation, download=True)

        train_set = [trainset]
        test_set  = [testset]
        num_classes = 10
        D = 32*32

    elif name == "MNIST":

        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                              transforms.ToTensor()]) 
        
        transform_validation = transforms.Compose([transforms.ToTensor()])

        train_set = [torchvision.datasets.MNIST(root=datapath+"MNIST", train=True, transform=transform_train, download=True)]
        test_set = [torchvision.datasets.MNIST(root=datapath+"MNIST", train=False, transform=transform_validation, download=True)]

        num_classes = 10
        D = 28*28
        
    elif name == "rotated_MNIST":

        angles = range(10,370,10)
        test_set = []
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                                transforms.ToTensor()])
        train_set = [torchvision.datasets.MNIST(root=datapath+"MNIST", train=True, transform=transform_train, download=True)]
        
        for a in angles:  
            transform_validation = transforms.Compose([transforms.RandomRotation((a,a+1)),
                                                transforms.ToTensor()])

            test_set_i = torchvision.datasets.MNIST(root=datapath+"MNIST", train=False, transform=transform_validation, download=True)
            test_set.append(test_set_i)

        num_classes = 10
        D = 28*28

    elif name == 'boston' or name == 'concrete' or name == 'airfoil' or name == 'elevators':
        datasets = ['boston', 'concrete', 'airfoil', 'elevators']
        for i,nm in enumerate(datasets):
            if nm == name:
                dataset_ind = i
        classnums = [1,1,1,1]
        num_classes = classnums[dataset_ind]
        n_splits=10 #number of splits in K-fold cross-validation
        # define the location of the dataset
        full_path = '{}{}'.format(datapath, name)
        
        Xo, Yo = load_UCIreg_dataset(full_path, name)
        
        # Shuffle data once before splitting
        N = Yo.shape[0]

        shuffle_ind = np.arange(0,N)
        random.shuffle(shuffle_ind)
        
        train_set = []
        test_set = []

        for split in range(n_splits):
            cut_start = split*int(np.floor(N/n_splits))
            cut_end = (split+1)*int(np.floor(N/n_splits))
            test_ind = shuffle_ind[cut_start:cut_end]
            train_ind = np.hstack((shuffle_ind[0:cut_start], shuffle_ind[cut_end:]))
            
            X_test = Xo[test_ind, :]
            y_test = Yo[test_ind]
            X_train = Xo[train_ind, :]
            y_train = Yo[train_ind]
        
            # Don't do it on the full data, but based on the training
            X_scaler = StandardScaler().fit(X_train)
            Y_scaler = StandardScaler().fit(y_train)
            X_train_norm = X_scaler.transform(X_train)
            X_test_norm = X_scaler.transform(X_test)
            y_train_norm = Y_scaler.transform(y_train)
            y_test_norm = Y_scaler.transform(y_test)

            D = X_train_norm.shape[1]

            X_train_t = torch.from_numpy(X_train_norm).float()
            X_test_t = torch.from_numpy(X_test_norm).float()

            y_train_t = torch.from_numpy(y_train_norm).float().squeeze()
            y_test_t = torch.from_numpy(y_test_norm).float().squeeze()
            
            train_set.append(UCI_dataset(X_train_t, y_train_t))
            test_set.append(UCI_dataset(X_test_t, y_test_t))
        
    else:
        datasets = ['diabetes', 'adult', 'connect-4', 'covtype']
        for i,nm in enumerate(datasets):
            if nm == name:
                dataset_ind = i
        classnums = [2,2,3,7]
        num_classes = classnums[dataset_ind]
        n_splits=10 #number of splits in K-fold cross-validation
        # define the location of the dataset
        full_path = '{}{}.csv'.format(datapath, name)
        
        # load the dataset
        X, y, cat_ix, num_ix = load_UCI_dataset(full_path)
        
        # Get categories for one-hot encoding for consistency accross folds
        enc = OneHotEncoder(handle_unknown='ignore')
        onehotenc = enc.fit(X)
        categories = [enc.categories_[i] for i in cat_ix.values]

        # define preprocessing steps (one-hot encoding + standard scaling)
        steps = [('c',OneHotEncoder(handle_unknown='ignore', categories = categories),cat_ix), ('n',StandardScaler(),num_ix)]

        # Generate transformer for one-hot encoding categorical and normalizing numerical features
        ct = ColumnTransformer(steps, sparse_threshold=0)
        N = y.shape[0] #Number of datapoints

        # Shuffle data once before splitting
        shuffle_ind = np.arange(0,N)
        random.shuffle(shuffle_ind)
        
        train_set = []
        test_set = []
        for split in range(n_splits): #Loop over the folds
            #Get train/test indexes for the current split
            cut_start = split*int(np.floor(N/n_splits))
            cut_end = (split+1)*int(np.floor(N/n_splits))
            test_ind = shuffle_ind[cut_start:cut_end]
            train_ind = np.hstack((shuffle_ind[0:cut_start], shuffle_ind[cut_end:]))

            X_test = X[test_ind, :]
            y_test = y[test_ind]
            X_train = X[train_ind, :]
            y_train = y[train_ind]

            norm = ct.fit(X_train) #Fit the preprocessing transform based on training data
            # Preprocess both train and test data with the fitted transform
            X_train_norm = norm.transform(X_train)
            X_test_norm = norm.transform(X_test)
            D = X_train_norm.shape[1] #Number of features
            
            # Transform to torch
            X_train_t = torch.from_numpy(X_train_norm).float()
            X_test_t = torch.from_numpy(X_test_norm).float()

            y_train_t = torch.from_numpy(y_train).long()
            y_test_t = torch.from_numpy(y_test).long()
            
            train_set.append(UCI_dataset(X_train_t, y_train_t))
            test_set.append(UCI_dataset(X_test_t, y_test_t))
    
    return train_set, test_set, num_classes, D
    
def UniformNoise(dataset, delta=1, train=False, size=2000, batch_size=None):
    if batch_size==None:
        batch_size=128
    import torch.utils.data as data_utils

    if dataset in ['MNIST', 'FMNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        shape = (3, 32, 32)
    elif dataset == 'diabetes':
        shape = (8,)
    elif dataset == 'adult':
        shape = (104,)
    elif dataset == 'connect-4':
        shape = (42,)
    elif dataset == 'covtype':
        shape = (54,)
    elif dataset == 'boston':
        shape = (12,)
    elif dataset == 'concrete':
        shape = (5,)
    elif dataset == 'airfoil':
        shape = (5,)
    elif dataset == 'elevators':
        shape = (18,)
    else:
        print("Dataset name not recognized")
        raise
    data = 2*delta*(torch.rand((size,) + shape) - 1/2)
    targets = torch.zeros(size)
    trainset = data_utils.TensorDataset(data, targets)
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader
    
