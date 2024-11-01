'''

module for checking support 


'''
import numpy as np 

import copy
import torch
from nflows import transforms, distributions, flows
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.auto import tqdm


class Support(object): 
    def __init__(self, device=None): 
        ''' Class for support. 
        '''
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.flow_support = None 
        self._flow = None 
    
    def check_support(self, X, Nsample=10000, threshold=0.95, return_support=False): 
        ''' check support for covariates X

        args: 
            X: NxD_x dimensional array

        kwargs: 
            Nsample: int specifying the number of samples.

            threshold: float between [0, 1], specifying the threshold. 

            return_support: bool, If True return the support percentiles instead of a boolean 
                array specifying whether X is in support. 
        '''
        # check that there is a flow 
        if self.flow_support is None: 
            raise ValueError('Either train or load the support flow that estimates p(X)') 

        X = np.atleast_2d(X)
        
        # check support
        with torch.no_grad(): 
            logpX = np.array(self.flow_support.log_prob(
                torch.tensor(X, dtype=torch.float32).to(self.device)).detach().cpu())
                
            _, logps = self.flow_support.sample_and_log_prob(Nsample)
            logps = np.array(logps.detach().cpu())
        
        support = np.zeros(X.shape[0])
        for i in range(X.shape[0]): 
            support[i] = np.mean(logpX[i] < logps)
            
        if return_support: 
            return support
        else: 
            return (support < threshold)

    def train(self, X, inverse_cdf=False, batch_size=50,
              learning_rate=5e-4, num_iter=300, clip_max_norm=5,
              verbose=False): 
        ''' Train a normalizing flow by providing the outcome, Y, and
        covariates, X. This function is a wrapper for `_train` and overwrites
        self.flow_support

        args: 
            Y: N x D_y array specifying the outcomes. N is the number of data
                points and D_y is the dimensionality of the outcome.

            X: N x D_x array specifying the covariates N is the number of data
                points and D_x is the dimensionality of the outcome.
        
        kwargs: 

            batch_size: int specifyin the training data batch size
                during training (Default: 50)

            learning_rate: float, specifying the learning rate used for
                training (Default: 5e-4) 

            clip_max_norm: float, clip gradients 

            verbose: bool, If True the function will print informative messages
                during the training.
        '''
        if self.flow_support is not None: 
            warnings.warn("Overwriting existing flow_support. clt+c if you don't want to do this")

        flow, _ = self._train(X, inverse_cdf=inverse_cdf, batch_size=batch_size,
                           num_iter=num_iter, learning_rate=learning_rate,
                           clip_max_norm=clip_max_norm, verbose=verbose)
        self.flow_support = flow
        return None 

    def load(self, file_flow):  
        ''' load flow that estimates p( X ), given the filename of the flow. 

        args: 
            file_flow: string specifying the directory path of the flow file 

        notes: 
        - ensembling not yet implemented
        '''
        if self.flow_support is not None: 
            warnings.warn("Overwriting existing flow_support. clt+c if you don't want to do this!")
        self.flow_support = torch.load(file_flow, map_location=self.device)
        return None 
    
    def load_optuna(self, study_name, study_dir, verbose=False):
        ''' load flow that estimates p( X ) best from an optuna study

        args: 
            study_name: str, name of optuna study

            study_dir: str, directory path where the optuna study is located

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 

        '''
        if self.flow_support is not None: 
            warnings.warn("Overwriting existing flow_support. clt+c if you don't want to do this!")
        import optuna 
        storage    = 'sqlite:///%s/%s/%s.db' % (study_dir, study_name, study_name)
        study = optuna.load_study(study_name=study_name, storage=storage)

        ntrial, values = [], [] 
        for trial in study.trials:
            if trial.values is not None:
                ntrial.append(trial.number)
                values.append(trial.values)
        if verbose: print('select best flow out of %i flows' % len(ntrial))

        ibest = np.array(ntrial)[np.argmin(np.concatenate(values))]

        # load flows with best loss. 
        fflow = '%s/%s/%s.%i.pt' % (study_dir, study_name, study_name, ibest)
        self.flow_support = torch.load(fflow, map_location=self.device)
        return None 

    def _train(self, X, inverse_cdf=False, batch_size=50, num_iter=300,
               learning_rate=5e-4, clip_max_norm=5, verbose=False): 
        ''' Train a normalizing flow by providing the outcome, Y, and
        covariates, X. 

        args: 
            Y: N x D_y array specifying the outcomes. N is the number of data
                points and D_y is the dimensionality of the outcome.

            X: N x D_x array specifying the covariates N is the number of data
                points and D_x is the dimensionality of the outcome.
        
        kwargs: 

            flow: flows.Flow object that specifies the architecture of the
                flow. See `set_architecture` function within the class for
                details.  
            
            batch_size: int specifyin the training data batch size
                during training (Default: 50)

            learning_rate: float, specifying the learning rate used for
                training (Default: 5e-4) 

            clip_max_norm: float, clip gradients 

            verbose: bool, If True the function will print informative messages
                during the training.
        '''
        X = np.atleast_2d(X)
        ndim = X.shape[1] 
        
        if inverse_cdf: raise NotImplementedError('inverse CDF transform is not yet implemented')
        # add any other checks we want to implement here. 

        # set up training/testing data
        Ntrain = int(0.9 * X.shape[0])
        if verbose: print('Ntrain= %i, Nvalid= %i' % (Ntrain, X.shape[0] - Ntrain))

        # shuffle up the data 
        np.random.seed(0)
        ind = np.arange(X.shape[0])
        np.random.shuffle(ind)

        data_train = X[ind][:Ntrain]
        data_valid = X[ind][Ntrain:]
        
        # set up data loaders
        train_loader = torch.utils.data.DataLoader(
                torch.tensor(data_train.astype(np.float32)).to(self.device),
                batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
                torch.tensor(data_valid.astype(np.float32)).to(self.device),
                batch_size=batch_size, shuffle=False)

        # training the flow 
        # specify the NDE architecture
        if self._flow is None: # arbitrary architecture
            self.set_architecture(ndim, nhidden=128, nblocks=5)

        flow = self._flow
        flow.to(self.device)

        # train flow
        optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

        best_epoch, best_valid_loss, valid_losses = 0, np.inf, []
        best_flow = None
        patience = 20 

        pbar = tqdm(range(num_iter), disable=(verbose == False), desc="Epoch")

        #for epoch in range(num_iter):
        for epoch in pbar: 
            # train 
            train_loss = 0.
            for batch in train_loader: 
                optimizer.zero_grad()
                loss = -flow.log_prob(batch).mean()
                loss.backward()
                train_loss += loss.item()
                clip_grad_norm_(flow.parameters(), max_norm=clip_max_norm)
                optimizer.step()
            train_loss = train_loss/float(len(train_loader))
        
            # validate
            with torch.no_grad():
                valid_loss = 0.
                for batch in valid_loader: 
                    loss = -flow.log_prob(batch).mean()
                    valid_loss += loss.item()
                valid_loss = valid_loss/len(valid_loader)

                if np.isnan(valid_loss): 
                    raise ValueError("NaN in validation loss. Check the input data. Or decrease clip_max_norm")

                valid_losses.append(valid_loss)
            
            if verbose: 
                pbar.set_description('TRAINING Loss: %.2e VALIDATION Loss: %.2e' % 
                                     (train_loss, valid_loss))
                
            if valid_loss < best_valid_loss: 
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_flow = copy.deepcopy(flow)
            else: 
                if epoch > best_epoch + patience: 
                    if verbose: 
                        pbar.set_description('DONE: BEST EPOCH %i BEST VALIDATION Loss: %.2e' %
                              (best_epoch, best_valid_loss))
                    break 

        if best_flow is None:
            raise ValueError("training failed") 
        best_flow.eval() 
        return best_flow, best_valid_loss

    def set_architecture(self, ndim, nhidden=128, nblock=5): 
        ''' make a flows.Flow object (i.e. a normalizing flow) q(X) by
        specifying the number of dimensions of X, number of hidden features,
        number of blocks. 
        
        At the moment it only supports Masked Autoregressive Flows
        (https://arxiv.org/abs/1912.02762) with a single multivariate gaussian
        as the base distribution.
        '''
        blocks = []
        for iblock in range(nblock): 
            blocks += [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=ndim, hidden_features=nhidden),
                    transforms.RandomPermutation(features=ndim)]
        transform = transforms.CompositeTransform(blocks)
    
        # single multivariate gaussian 
        base_distribution = distributions.StandardNormal(shape=[ndim])

        self._flow = flows.Flow(transform=transform, distribution=base_distribution)
        return None 
