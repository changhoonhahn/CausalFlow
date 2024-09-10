'''


main causalflow module 


# todo
- if we want to implement inverse CDF, the easiest solution will probably be to
  implement a tailored flow class, where all the functions inverse CDF and CDF
  transform the data. 

'''
import numpy as np 
import warnings

import torch
from torch import nn

from sbi import utils as Ut
from sbi import inference as Inference


class BaseCausalFlow(object): 
    def __init__(self, device=None): 
        ''' Base CausalFlow class 
        '''
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def _train_flow(self, Y, X, outcome_range=[None, None], inverse_cdf=False,
                   nde=None, training_batch_size=50, learning_rate=5e-4,
                   verbose=False): 
        ''' Train a normalizing flow by providing the outcome, Y, and
        covariates, X. 

        args: 
            Y: N x D_y array specifying the outcomes. N is the number of data
                points and D_y is the dimensionality of the outcome.
            X: N x D_x array specifying the covariates N is the number of data
                points and D_x is the dimensionality of the outcome.
        
        kwargs: 
            outcome_range: tuple specifying the range of the outcomes. This
                should be some physically meaningful limit. For example, if
                outcome is dollar amount, then a lower bound of 0 is necessary.
                Otherwise set it to very large amounts. 
            inverse_cdf: If True, implement an inverse CDF transform on the
                dataset. 
            nde: sbi.utils.posterior_nn object that specifies the architecture
                of the nde.  
            training_batch_size: int specifyin the training data batch size
                during training (Default: 50)
            learning_rate: float, specifying the learning rate used for
                training (Default: 5e-4) 
            verbose: bool, If True the function will print informative messages
                during the training.
        '''
        # check outcome range
        if outcome_range[0] is None or outcome_range[1] is None: 
            raise ValueError('set range for the outcomes.')   

        # check dimensionality of outcome and covariates 
        Y = np.atleast_2d(Y)
        X = np.atleast_2d(X)
        
        if Y.shape[0] != X.shape[0]: 
            raise ValueError('there is a dimensionality mismatch between the outcome and the coviarates')
        # check that dimensionity of outcome_range is consistent with D_y 
        if Y.shape[1] != len(np.array(outcome_range[0])) or Y.shape[1] != len(np.array(outcome_range[1])):
            raise ValueError('there is a dimensionality mismatch between the outcome and the outcome range')

        if inverse_cdf: raise NotImplementedError('inverse CDF transform is not yet implemented')
        # add any other checks we want to implement here. 


        # training the flow 
        # set prior 
        lower_bounds = torch.tensor(outcome_range[0]).to(self.device)
        upper_bounds = torch.tensor(outcome_range[1]).to(self.device)
        prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=self.device)
    
        # specify the NDE architecture
        if nde is None: 
            # default NDE architecture: Masked Autoencoder for Distribution
            # Estimation (https://arxiv.org/abs/1502.03509) with arbitrary
            # number of hidden features and blocks. 
            nde = Ut.posterior_nn('made', 
                    hidden_features=64, 
                    num_transforms=2, 
                    num_blocks=2, 
                    num_mixture_components=2,
                    use_batch_norm=True)

        # set up the SNPE object
        anpe = Inference.SNPE(
                    prior=prior,
                    density_estimator=nde,
                    device=self.device)
        
        # load training data 
        anpe.append_simulations(
                torch.as_tensor(Y.astype(np.float32)).to(self.device),
                torch.as_tensor(X.astype(np.float32)).to(self.device))
    
        # train!
        p_x_y_est = anpe.train(
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,
                show_train_summary=verbose)

        flow = anpe.build_posterior(p_x_y_est)
        return flow 

    def _load_flow(self, _dir, n_ensemble=5, flow_name=None):
        ''' given directory load flows. By default it will load an ensemble of
        flows based on an optuna study. 

        args: 
            _dir: string specifying the directory path of the flows  

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 
            flow_name: (str) specify the filename of the flow. If `flow_name`
                is specified, it will read that flow. 
        '''
        # try to load optuna study to determine the n_ensemble best flows. 

        qphis = []
        for fqphi in glob.glob('%s/*.pt' % _dir):
            qphi = torch.load(fqphi, map_location=device)
            qphis.append(qphi)
        return qphis


class CausalFlowA(BaseCausalFlow): 
    def __init__(self, device=None): 
        ''' CausalFlow for Scenario A, where you're using flows to estimate
        both $p( Y | X, T=1 )$ and $p( Y | X, T=0 )$.

        explain scenario A

        explain the algorithm 
        
        explain how to use it 


        Notes: 
        * Either train the treated/control flows or load them. 
        '''
        self.flow_treated = None # flow estimating p( Y | X, T=1 )
        self.flow_control = None # flow estimating p( Y | X, T=0 )

        super().__init__(device=device)
    
    def CATE(self, X, Nsample=10000, progress_bar=False): 
        ''' evaluate the conditional average treatment effect for a given
        covariate value, X


        CATE = int Y p(Y|X,T=1) dY - int Y p(Y|X,T=0) dY

        args: 
            X: 1D array with D_x dimensions. The given covariate value to
                estimate the CATE. 
        
        kwargs: 
            Nsample: int, the number of samples used for the Monte Carlo
                integration estimates
        '''
        # sample p( Y | X, T=1 ) 
        Ys_treated = self._sample(self.flow_treated, X, Nsample=Nsample,
                                  progress_bar=progress_bar)
        # sample p( Y | X, T=0 ) 
        Ys_control = self._sample(self.flow_control, X, Nsample=Nsample,
                                  progress_bar=progress_bar)
        # estimate the cate 
        cate = np.mean(Ys_treated) - np.mean(Ys_control) 
        return cate
    
    def train_flow_treated(self, Y_treat, X_treat, **kwargs):
        ''' Train a flow for the treated sample to estimate p( Y | X, T=1 ).
        This is a wrapper for self._train_flow. See self._train_flow for
        documentation and detail on training. 
        '''
        if self.flow_treated is not None: 
            warnings.warn("Overwriting existing flow_treated. clt+c if you
                          don't want to do this")

        self.flow_treated = self._train_flow(self, Y_treat, X_treat, **kwargs) 
        return None  
    
    def train_flow_control(self, Y_cont, X_cont, **kwargs):
        ''' Train a flow for the treated sample to estimate p( Y | X, T=0 ). 
        This is a wrapper for self._train_flow. See self._train_flow for
        documentation and detail on training. 
        '''
        if self.flow_control is not None: 
            warnings.warn("Overwriting existing flow_treated. clt+c if you
                          don't want to do this")

        self.flow_control = self._train_flow(self, Y_cont, X_cont, **kwargs) 
        return None  

    def load_flow_treated(self, _dir, n_ensemble=5, flow_name=None): 
        ''' load flow that estimates p( Y | X, T=1 ), given either a directory
        or filename of the flow. By default it will load an ensemble of
        flows based on an optuna study. 

        args: 
            _dir: string specifying the directory path of the treated flows  

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 
            flow_name: (str) specify the filename of the flow. If `flow_name`
                is specified, it will read that flow. 
        '''
        self.flow_treated = self._load_flow(_dir, n_ensemble=5, flow_name=None)
        return None 

    def load_flow_control(self, _dir, n_ensemble=5, flow_name=None): 
        ''' load flow that estimates p( Y | X, T=0 ), given either a directory
        or filename of the flow. By default it will load an ensemble of
        flows based on an optuna study. 

        args: 
            _dir: string specifying the directory path of the control flows  

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 
            flow_name: (str) specify the filename of the flow. If `flow_name`
                is specified, it will read that flow. 
        '''
        self.flow_treated = self._load_flow(_dir, n_ensemble=5, flow_name=None)
        return None 

    def _sample(self, flow, X, Nsample=10000, progress_bar=False):
        ''' sample given flow at given covariate X: Y' ~ flow( Y | X ). This is
        used for calculating the conditional average treatment effect but can
        also be used to examine the overall distribution of outcomes for the
        given flow. 

        args: 
            flow: flow object  
            X: D_x 1D array specifying the covariate values to sample and
                evaluate
        
        kwargs: 
            Nsample: int, specifying the number of samples to draw. More samples
                will increase the fidelity of estimating the outcome
                distribution. 
            progress_bar: bool, If True it will show a nice progress bar so it
                looks like it's doing something
        '''
        # sample the flow 
        Yp = flow.sample((Nsample,), 
                         x=torch.as_tensor(X).to(self.device), # specify covariate
                         show_progress_bars=progress_bar)
        Yp = Yp.detach().cpu().numpy()
        return Yp


class CausalFlowB(BaseCausalFlow): 
    def __init__(self, ): 
        ''' CausalFlow for Scenario B, where you're using flows to estimate
        only $p( Y | X, T=1 )$ or $p( Y | X, T=0 )$, but not both. This method
        is equivalent to ``synthetic matching''. 
        '''
        super().__init__()


class CausalFlowC(BaseCausalFlow): 
    def __init__(self, ): 
        ''' CausalFlow for Scenario C, where you're using a flow to estimate
        $p( Y | X, T )$. This is for the scenario where treatment is not
        binary but continuous.  


        The treatment effect is estimated by comparing 
        $p( Y | X_fid, T' )$ and $p( Y | X_fid, T_fid )$. 
        X_fid a fiducial value of the covariates. T_fid is the fiducial
        treatment level, while T' is T_fid + dT where dT is the ``treatment''.
        '''

