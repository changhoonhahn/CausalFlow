''' 

script for training a large number of flows using the Optuna hyperparameter
optimization framework. 

'''
import os,sys
import numpy as np
from astropy.table import Table
from causalflow import causalflow

import torch
import optuna 

##################################################################################
treat_or_control    = sys.argv[1]
nf_model            = sys.argv[2]
study_name          = sys.argv[3]
output_dir          = sys.argv[4]
##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
# read dataset 
fema = Table.read('../dat/cano2024_dataset.hdf5')
control = (fema['communityRatingSystemDiscount'] == 11.)
treated = ~control

# read in training data 
columns = ['amountPaidOnTotalClaim_per_policy', # outcome
           'mean_rainfall', 'avg_risk_score_all', 'median_household_income', 'population', # covariates
           'renter_fraction', 'educated_fraction', 'white_fraction']
if treat_or_control == 'treated': 
    train_data = np.vstack([np.ma.getdata(fema[col].data) for col in columns]).T[treated]
elif treat_or_control == 'control': 
    train_data = np.vstack([np.ma.getdata(fema[col].data) for col in columns]).T[control]
else: 
    raise ValueError

# reduce dynamical range of some of the columns
train_data[:,0] = np.log10(train_data[:,0])
train_data[:,3] = np.log10(train_data[:,3])
train_data[:,4] = np.log10(train_data[:,4])
##################################################################################
# OPTUNA
##################################################################################
# declare Scenario A CausalFlow
Cflow = causalflow.CausalFlowA(device=device)

# Optuna Parameters
n_trials    = 1000
n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5 
n_transf_min, n_transf_max = 2, 5 
n_hidden_min, n_hidden_max = 32, 128 
n_comp_min, n_comp_max = 1, 5
n_lr_min, n_lr_max = 5e-6, 1e-3 


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 

    if nf_model == 'made': 
        n_comp = trial.suggest_int("n_comp", n_comp_min, n_comp_max)
    else: 
        n_comp = 1

    Cflow.set_architecture(
            arch=nf_model, 
            nhidden=n_hidden, 
            ntransform=n_transf, 
            nblocks=n_blocks,
            num_mixture_components=n_comp, 
            batch_norm=True)

    flow, best_valid_log_prob = Cflow._train_flow(train_data[:,0], train_data[:,1:], 
            outcome_range=[[-1.], [6.]], 
            training_batch_size=50, 
            learning_rate=lr, 
            verbose=False)

    # save trained NPE  
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flow, fflow)

    return -1*best_valid_log_prob

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) 
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) 

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
