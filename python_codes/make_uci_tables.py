# A file for generating .tex tables of calculated and saved UCI classification task results.

import numpy as np
from sys import argv
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel

def load_results(dataset, setup):
    # Load results
    n_splits = 10
    predlist = []
    labelslist = []
    for i in range(n_splits):
        results = np.load('./{}{}/KFAC_result_{}.npz'.format(dataset,setup,i))
        preds = results['predictions']
        labels = results['targets']
        predlist.append(preds)
        labelslist.append(labels)

    NLLLosses = np.zeros(n_splits)
    accuracies = np.zeros(n_splits)
    AUCs = np.zeros(n_splits)

    for i in range(n_splits):
        x = predlist[i]
        y = labelslist[i].astype(np.compat.long)
        NLLLosses[i] = F.nll_loss(torch.from_numpy(np.log(x)), torch.from_numpy(y))
        preds = np.argmax(x, axis=-1)
        hits = np.sum(y.astype(int) == preds)
        accuracies[i] = hits/y.shape[0]

        num_classes = x.shape[1]
        AUC = np.zeros(num_classes)
        for c in range(num_classes):
            AUC[c] = roc_auc_score(y==c, x[:,c])
        AUCs[i] = np.mean(AUC)
    
    return NLLLosses, accuracies, AUCs

def print_table(methods, data_names, minibatch_size, D, N, C, table_contents, min_pval = 0.01):

    dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
    method_names = ['ReLU', 'loc RBF', 'glob RBF (sin)', 'glob RBF (tri)', 'loc Mat-5/2', 'glob Mat-5/2 (sin)' ,'glob Mat-5/2 (tri)' ,'loc Mat-3/2' ,'glob Mat-3/2 (sin)' ,'glob Mat-3/2 (tri)' , 'glob RBF (sincos)', 'glob Mat-5/2 (sincos)', 'glob Mat-3/2 (sincos)', 'glob RBF (prelu)', 'glob Mat-5/2 (prelu)', 'glob Mat-3/2 (prelu)']
    
    print('\n\n%{} (this table has been automatically generated, do not alter manually)'.format(table_contents))

    if table_contents == 'NLPD_ACC':
        # Column labels (row0)
        print('{:<25} '.format(''),end='')
        for data_name in data_names:
            print('& {:^45} '.format('\\multicolumn{2}{c}{\sc '+data_name+'}'),end='')
        print('\\\\')
        
        # Column labels (row1)
        print('{:<25} '.format('($n$, $d$)'),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^45} '.format('\\multicolumn{{2}}{{c}}{{({}, {})}}'.format(N[j],D[j])),end='')
        print('\\\\')
        
        # Column labels (row2)
        print('{:<25} '.format('($c$, $n_\\textrm{batch}$)'),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^45} '.format('\\multicolumn{{2}}{{c}}{{({}, {})}}'.format(C[j],minibatch_size[j])),end='')
        print('\\\\')
        print('\midrule')
        
        # Column labels (row3)
        print('{:<25} '.format(''),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^21} '.format('NLPD'),end='')
            print('& {:^21} '.format('ACC'),end='')

            
    elif table_contents == 'AUC':
        # Column labels (row0)
        print('{:<25} '.format(''),end='')
        for data_name in data_names:
            print('& {:^45} '.format('{\sc '+data_name.replace("_", "\\_")+'}'),end='')
        print('\\\\')
        
        # Column labels (row1)
        print('{:<25} '.format('($n$, $d$)'),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^45} '.format('({}, {})'.format(N[j],D[j])),end='')
        print('\\\\')
        
        # Column labels (row2)
        print('{:<25} '.format('($c$, $n_\\textrm{batch}$)'),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^45} '.format('({}, {})'.format(C[j],minibatch_size[j])),end='')
        print('\\\\')
        print('\midrule')
        
        # Column labels (row3)
        print('{:<25} '.format(''),end='')
        for j,data_name in enumerate(data_names):
            print('& {:^45} '.format('AUC'),end='')
            
    print('\\\\')
    print('\midrule')
    
    # Precalculate table values to find best values for bolding
    table_ACC = np.zeros((len(methods), len(data_names), 10))
    table_NLPD = np.zeros((len(methods), len(data_names), 10))
    table_AUC = np.zeros((len(methods), len(data_names), 10))
    
    for i,m in enumerate(methods):
        for j in range(len(data_names)):
            try:
                NLLLosses, accuracies, AUC = load_results(data_names[j], dir_names[m])
            except:
                NLLLosses, accuracies, AUC = 1000*np.ones(10), np.zeros(10), np.zeros(10)
            table_ACC[i,j,:] = accuracies
            table_NLPD[i,j,:] = NLLLosses
            table_AUC[i,j,:] = AUC
    
    # Find best value index for each dataset and each metric     
    acc_bold_inds = []
    nlpd_bold_inds = []
    auc_bold_inds = []
    table_ACC[np.isnan(table_ACC)] = 0
    table_AUC[np.isnan(table_AUC)] = 0
    table_NLPD[np.isnan(table_NLPD)] = 1000
    
    table_ACC_mean = np.mean(table_ACC, axis=-1)
    table_ACC_std = np.std(table_ACC, axis=-1)
    table_AUC_mean = np.mean(table_AUC, axis=-1)
    table_AUC_std = np.std(table_AUC, axis=-1)
    table_NLPD_mean = np.mean(table_NLPD, axis=-1)
    table_NLPD_std = np.std(table_NLPD, axis=-1)
    
    for j in range(len(data_names)):
        acc_bold_inds.append([]) 
        nlpd_bold_inds.append([])
        auc_bold_inds.append([])
        acc_bold_inds[-1].append(np.argmax(table_ACC_mean[:,j]))
        nlpd_bold_inds[-1].append(np.argmin(table_NLPD_mean[:,j]))
        auc_bold_inds[-1].append(np.argmax(table_AUC_mean[:,j]))
    
    for j in range(len(data_names)):

        argmax_acc_ = acc_bold_inds[j][0]
        argmin_nlpd_ = nlpd_bold_inds[j][0]
        argmax_auc_ = auc_bold_inds[j][0]

        max_acc_ = table_ACC[argmax_acc_, j, :]
        min_nlpd_ = table_NLPD[argmin_nlpd_, j, :]
        max_auc_ = table_AUC[argmax_auc_, j, :]

        for i in range(len(methods)):
            if i != argmax_acc_:
                tStat, pValue = ttest_rel(max_acc_, table_ACC[i,j,:])
                if pValue > min_pval:
                    acc_bold_inds[j].append(i)
            
            if i != argmin_nlpd_:
                tStat, pValue = ttest_rel(min_nlpd_, table_NLPD[i,j,:])
                if pValue > min_pval:
                    nlpd_bold_inds[j].append(i)
            
            if i != argmax_auc_:
                tStat, pValue = ttest_rel(max_auc_, table_AUC[i,j,:])
                if pValue > min_pval:
                    auc_bold_inds[j].append(i)
    
    # The rest of the rows
    for k,m in enumerate(methods):

        print('{:<25} '.format(method_names[m]),end='')
        for j,data_name in enumerate(data_names):
            bf_acc = ''
            bf_nlpd = ''
            bf_auc = ''
            found = True
            if table_ACC_mean[k,j] == 0:
                found = False
            if k == acc_bold_inds[j][0]:
                bf_acc = '\\bf '
            if k == nlpd_bold_inds[j][0]:
                bf_nlpd = '\\bf '
            if k == auc_bold_inds[j][0]:
                bf_auc = '\\bf '
            if found == True:
                if table_contents == 'NLPD_ACC':
                    print('& {:^21} '.format('${}{:.2f}{{\pm}}{:.2f}$'.format(bf_nlpd,table_NLPD_mean[k,j],table_NLPD_std[k,j])),end='')
                    print('& {:^21} '.format('${}{:.2f}{{\pm}}{:.2f}$'.format(bf_acc,table_ACC_mean[k,j],table_ACC_std[k,j])),end='')
                elif table_contents == 'AUC':
                    print('& {:^45} '.format('${}{:.2f}{{\pm}}{:.2f}$'.format(bf_auc,table_AUC_mean[k,j],table_AUC_std[k,j])),end='')
            if found==False:
                if table_contents == 'NLPD_ACC':
                    print('& {:^21} '.format('---'),end='')
                    print('& {:^21} '.format('---'),end='')
                elif table_contents == 'AUC':
                    print('& {:^45} '.format('---'),end='')
        print('\\\\')

## FULL TABLE
def full_table(table_contents):

    # What to show?
    methods = range(16)
    data_names = ['diabetes', 'adult', 'connect-4', 'covtype']
    minibatch_size = [50,500,500,500]
    D = [8, 14, 42, 54]
    N = [768, 45222, 67556, 581912]
    C = [2,2,3,7]

    print_table(methods, data_names, minibatch_size, D, N, C, table_contents)

## SHORT TABLE
def short_table(table_contents):

    # What to show?
    methods = [0,1,2,3,10,13]
    data_names = ['diabetes', 'adult', 'connect-4', 'covtype']
    minibatch_size = [50,500,500,500]
    D = [8, 14, 42, 54]
    N = [768, 45222, 67556, 581912]
    C = [2,2,3,7]

    print_table(methods, data_names, minibatch_size, D, N, C, table_contents)

# Handle options
table_size = str(argv[1])
table_contents = str(argv[2])
if table_size=='full':
    full_table(table_contents)
elif table_size=='short':
    short_table(table_contents)
else:
    print('Argumets: full/short NLPD_ACC/AUC')
