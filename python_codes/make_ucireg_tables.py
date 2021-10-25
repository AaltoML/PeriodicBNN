# A file for generating .tex tables of calculated and saved UCI regression task results.

import numpy as np
from sys import argv
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.stats import ttest_rel

def load_results(dataset, setup):
    # Load results
    n_splits = 10
    predlist = []
    labelslist = []
    predvar_list = []
    s_list = []
    for i in range(n_splits):
        results = np.load('./{}{}/KFAC_result_{}.npz'.format(dataset,setup,i))
        preds = results['predictions']
        labels = results['targets']
        predvar = results['predictive_variance']
        s = results['s']
        predlist.append(preds)
        labelslist.append(labels)
        predvar_list.append(predvar)
        s_list.append(s)

    NLLLosses = np.zeros(n_splits)
    RMSEs = np.zeros(n_splits)

    for i in range(n_splits):
        x = predlist[i].squeeze()
        y = labelslist[i].squeeze()
        var = predvar_list[i].squeeze()
        s = s_list[i]
        NLLLosses[i] = -1*np.mean(norm.logpdf(x - y, loc=0, scale=np.sqrt(s**2 + var)))
        RMSEs[i] = np.sqrt(np.mean((x - y)**2))
    
    return NLLLosses, RMSEs

def print_table(methods, data_names, minibatch_size, D, N, C, min_pval = 0.01):

    dir_names = ['relu','RBF_local','RBF_sin','RBF_tri','matern52_local','matern52_sin','matern52_tri','matern32_local','matern32_sin','matern32_tri', 'RBF_sincos', 'matern52_sincos', 'matern32_sincos', 'RBF_prelu', 'matern52_prelu', 'matern32_prelu']
    method_names = ['ReLU', 'loc RBF', 'glob RBF (sin)', 'glob RBF (tri)', 'loc Mat-5/2', 'glob Mat-5/2 (sin)' ,'glob Mat-5/2 (tri)' ,'loc Mat-3/2' ,'glob Mat-3/2 (sin)' ,'glob Mat-3/2 (tri)' , 'glob RBF (sincos)', 'glob Mat-5/2 (sincos)', 'glob Mat-3/2 (sincos)', 'glob RBF (prelu)', 'glob Mat-5/2 (prelu)', 'glob Mat-3/2 (prelu)']
    
    print('\n\n%NLPD and RMSE (this table has been automatically generated, do not alter manually)')

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
        print('& {:^21} '.format('RMSE'),end='')
            
    print('\\\\')
    print('\midrule')
    
    # Precalculate table values to find best values for bolding
    table_RMSE = np.zeros((len(methods), len(data_names), 10))
    table_NLPD = np.zeros((len(methods), len(data_names), 10))
    
    for i,m in enumerate(methods):
        for j in range(len(data_names)):
            try:
                NLLLosses, RMSEs = load_results(data_names[j], dir_names[m])
            except:
                NLLLosses, RMSEs = 1000*np.ones(10), 1000*np.ones(10)
            table_RMSE[i,j,:] = RMSEs
            table_NLPD[i,j,:] = NLLLosses
    
    # Find best value index for each dataset and each metric     
    rmse_bold_inds = []
    nlpd_bold_inds = []
    table_RMSE[np.isnan(table_RMSE)] = 1000
    table_NLPD[np.isnan(table_NLPD)] = 1000
    
    table_RMSE_mean = np.mean(table_RMSE, axis=-1)
    table_RMSE_std = np.std(table_RMSE, axis=-1)
    table_NLPD_mean = np.mean(table_NLPD, axis=-1)
    table_NLPD_std = np.std(table_NLPD, axis=-1)

    for j in range(len(data_names)):
        rmse_bold_inds.append([]) 
        nlpd_bold_inds.append([]) 
        rmse_bold_inds[-1].append(np.argmin(table_RMSE_mean[:,j]))
        nlpd_bold_inds[-1].append(np.argmin(table_NLPD_mean[:,j]))

    for j in range(len(data_names)):

        argmin_rmse_ = rmse_bold_inds[j][0]
        argmin_nlpd_ = nlpd_bold_inds[j][0]

        min_rmse_ = table_RMSE[argmin_rmse_, j, :]
        min_nlpd_ = table_NLPD[argmin_nlpd_, j, :]

        for i in range(len(methods)):
            if i != argmin_rmse_:
                tStat, pValue = ttest_rel(min_rmse_, table_RMSE[i,j,:])
                if pValue > min_pval:
                    rmse_bold_inds[j].append(i)

            if i != argmin_nlpd_:
                tStat, pValue = ttest_rel(min_nlpd_, table_NLPD[i,j,:])
                if pValue > min_pval:
                    nlpd_bold_inds[j].append(i)

    
    # The rest of the rows
    for k,m in enumerate(methods):

        print('{:<25} '.format(method_names[m]),end='')
        for j,data_name in enumerate(data_names):
            bf_rmse = ''
            bf_nlpd = ''
            found = True
            if table_RMSE_mean[k,j] == 1000:
                found = False
            if k == rmse_bold_inds[j][0]:
                bf_rmse = '\\bf '
            if k == nlpd_bold_inds[j][0]:
                bf_nlpd = '\\bf '
            if found == True:
                print('& {:^21} '.format('${}{:.2f}{{\pm}}{:.2f}$'.format(bf_nlpd,table_NLPD_mean[k,j],table_NLPD_std[k,j])),end='')
                print('& {:^21} '.format('${}{:.2f}{{\pm}}{:.2f}$'.format(bf_rmse,table_RMSE_mean[k,j],table_RMSE_std[k,j])),end='')
            if found==False:
                print('& {:^21} '.format('---'),end='')
                print('& {:^21} '.format('---'),end='')
        print('\\\\')

## FULL TABLE
def full_table():

    # What to show?
    methods = range(16)
    data_names = ['boston', 'concrete', 'airfoil', 'elevators']
    minibatch_size = [50,50,50,500]
    D = [12,5,5,18]
    N = [506,1030,1503,16599]
    C = [1,1,1,1]

    print_table(methods, data_names, minibatch_size, D, N, C)

## SHORT TABLE
def short_table():

    # What to show?
    #methods = [0,1,2,10,7,8,12]
    methods = [0,1,2,13,7,8,15]
    data_names = ['boston', 'concrete', 'airfoil', 'elevators']
    minibatch_size = [50,50,50,500]
    D = [12,5,5,18]
    N = [506,1030,1503,16599]
    C = [1,1,1,1]

    print_table(methods, data_names, minibatch_size, D, N, C)

# Handle options
table_size = str(argv[1])
if table_size=='full':
    full_table()
elif table_size=='short':
    short_table()
else:
    print('Argumets: full/short')
