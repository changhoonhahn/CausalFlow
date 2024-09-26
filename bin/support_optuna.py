''' 

script for training a large number of support flows using the Optuna hyperparameter
optimization framework. 

'''
import os,sys
import numpy as np
from astropy.table import Table
from causalflow import support as Support

import torch
import optuna 

##################################################################################
treat_or_control    = sys.argv[1]
study_name          = sys.argv[2]
output_dir          = sys.argv[3]
##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################################
# read dataset 
fema = Table.read('../dat/cano2024_dataset.hdf5')
control = (fema['communityRatingSystemDiscount'] == 11.)
treated = ~control

# read in training data 
columns = ['mean_rainfall', 'avg_risk_score_all', 'median_household_income', 'population', # covariates
           'renter_fraction', 'educated_fraction', 'white_fraction']
if treat_or_control == 'treated': 
    train_data = np.vstack([np.ma.getdata(fema[col].data) for col in columns]).T[treated]
elif treat_or_control == 'control': 
    train_data = np.vstack([np.ma.getdata(fema[col].data) for col in columns]).T[control]
else: 
    raise ValueError

# reduce dynamical range of some of the columns
train_data[:,3] = np.log10(train_data[:,3])
train_data[:,4] = np.log10(train_data[:,4])
ndim = train_data.shape[1]
##################################################################################
# OPTUNA
##################################################################################
# declare Scenario A CausalFlow
Sup = Support.Support(device=device) 

# Optuna Parameters
n_trials    = 1000
n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5 
n_hidden_min, n_hidden_max = 32, 128 
n_lr_min, n_lr_max = 5e-6, 1e-3 


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 

    # set architecture
    Sup.set_architecture(ndim, 
            nhidden=n_hidden, 
            nblock=n_blocks) 
    
    # run trianing
    flow, best_valid_loss = Sup.train(train_data, 
            batch_size=50, 
            learning_rate=lr, 
            num_iter=300,
            clip_max_norm=1, 
            verbose=False)

    # save trained flow  
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flow, fflow)

    return best_valid_loss

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) 
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) 

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
