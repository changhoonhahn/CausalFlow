'''

module for checking support 


'''
import numpy as np 
import warning 

import copy
import torch
from nflows import transforms, distributions, flows
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_


class Support(object): 
    def __init__(self, device=None): 
        ''' Class for support. 
        '''
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.flow_support = None 
    
    def check_support(self, X, nsample=10000, threshold=0.95): 
        ''' check support for covariates X

        args: 
            X: NxD_x dimensional array

        kwargs: 
            nsample: int specifying the number of samples.

            threshold: float between [0, 1], specifying the threshold. 
        '''
        # check that there is a flow 
        if self.flow_support is None: 
            raise ValueError('Either train or load the support flow that estimates p(X)') 

        X = np.atleast_2d(X)
        
        # check support
        with torch.no_grad(): 
            logpX = np.array(self.flow_support.log_prob(
                torch.tensor(X, dtype=torch.float32).to(device)).detach().cpu())
                
            _, logps = self.flow_support.sample_and_log_prob(nsample)
            logps = np.array(logps.detach().cpu())
        
        support = np.zeros(X.shape[0])
        for i in range(X.shape[0]): 
            support[i] = np.mean(logpX[i] < logps)
            
        return (support < threshold)

    def train_support_flow(self, X, inverse_cdf=False,
                            flow=None, batch_size=50, learning_rate=5e-4,
                            clip_max_norm=5, verbose=False): 
        ''' Train a normalizing flow by providing the outcome, Y, and
        covariates, X. This function is a wrapper for `_train_support_flow` and
        overwrites self.flow_support

        args: 
            Y: N x D_y array specifying the outcomes. N is the number of data
                points and D_y is the dimensionality of the outcome.

            X: N x D_x array specifying the covariates N is the number of data
                points and D_x is the dimensionality of the outcome.
        
        kwargs: 

            flow: flows.Flow object that specifies the architecture of the
                flow. See `_make_flow` function within the class for details.  
            
            batch_size: int specifyin the training data batch size
                during training (Default: 50)

            learning_rate: float, specifying the learning rate used for
                training (Default: 5e-4) 

            clip_max_norm: float, clip gradients 

            verbose: bool, If True the function will print informative messages
                during the training.
        '''
        if self.flow_support is not None: 
            warnings.warn("Overwriting existing flow_support. clt+c if you
                          don't want to do this")

        flow = self._train_support_flow(X, inverse_cdf=inverse_cdf,
                                        flow=flow, batch_size=batch_size,
                                        learning_rate=learning_rate,
                                        clip_max_norm=clip_max_norm,
                                        verbose=verbose)
        self.flow_support = flow
        return None 

    def load_support_flow(self, file_flow):  
        ''' load flow that estimates p( X ), given the filename of the flow. 

        args: 
            file_flow: string specifying the directory path of the flow file 

        notes: 
        - ensembling not yet implemented
        '''
        if self.flow_support is not None: 
            warnings.warn("Overwriting existing flow_support. clt+c if you
                          don't want to do this!")
        self.support_flow = torch.load(file_flow, map_location=self.device)
        return None 

    def _train_support_flow(self, X, inverse_cdf=False,
                            flow=None, batch_size=50, learning_rate=5e-4,
                            clip_max_norm=5, verbose=False): 
        ''' Train a normalizing flow by providing the outcome, Y, and
        covariates, X. 

        args: 
            Y: N x D_y array specifying the outcomes. N is the number of data
                points and D_y is the dimensionality of the outcome.

            X: N x D_x array specifying the covariates N is the number of data
                points and D_x is the dimensionality of the outcome.
        
        kwargs: 

            flow: flows.Flow object that specifies the architecture of the
                flow. See `_make_flow` function within the class for details.  
            
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
        if verbose: print('Ntrain= %i, Nvalid= %i' % (Ntrain, data.shape[0] - Ntrain))

        # shuffle up the data 
        np.random.seed(0)
        ind = np.arange(data.shape[0])
        np.random.shuffle(ind)

        data_train = X[ind][:Ntrain]
        data_valid = X[ind][Ntrain:]
        
        # set up data loaders
        train_loader = torch.utils.data.DataLoader(
                torch.tensor(data_train.astype(np.float32)).to(device),
                batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
                torch.tensor(data_valid.astype(np.float32)).to(device),
                batch_size=batch_size, shuffle=False)

        # training the flow 
        # specify the NDE architecture
        if flow is None: 
            # arbitrary architecture
            flow = self._make_flow(ndim, 128, 5)
            flow.to(self.device)

        # train flow
        optimizer = optim.Adam(flow.parameters(), lr=learning_rate)

        best_epoch, best_valid_loss, valid_losses = 0, np.inf, []
        best_flow = None
        patience = 20 
        for epoch in range(num_iter):
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
                    raise ValueError("NaN in validation loss. Check the input
                                     data. Or try running the training again")

                valid_losses.append(valid_loss)
            
            if verbose and (epoch % 10 == 0): 
                print('Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' %
                      (epoch, train_loss, valid_loss))
                
            if valid_loss < best_valid_loss: 
                best_valid_loss = valid_loss
                best_epoch = epoch
                best_flow = copy.deepcopy(flow)
            else: 
                if epoch > best_epoch + patience: 
                    if verbose: 
                        print('DONE: EPOCH %i, BEST EPOCH %i BEST VALIDATION Loss: %.2e' %
                              (epoch, best_epoch, best_valid_loss))
                    break 

        if best_flow is None:
            raise ValueError("training failed") 

        return best_flow 

    def _make_flow(self, ndim, nhidden, nblock):
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

        flow = flows.Flow(transform=transform, distribution=base_distribution)
        return flow 
