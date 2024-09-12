'''


main causalflow module 


# todo
- if we want to implement inverse CDF, the easiest solution will probably be to
  implement a tailored flow class, where all the functions inverse CDF and CDF
  transform the data. 

- implement validation functions in BaseCausalFlow 

'''
import numpy as np 
import warnings

import torch
from torch import nn

from sbi import utils as Ut
from sbi import inference as Inference

from . import support as Support


class BaseCausalFlow(object): 
    def __init__(self, device=None): 
        ''' Base CausalFlow class 
        '''
        if device is None:
            if torch.cuda.is_available(): self.device = 'cuda'
            else: self.device = 'cpu'

        self._nde = None 

    def _train_flow(self, Y, X, outcome_range=[None, None], inverse_cdf=False,
                    training_batch_size=50, learning_rate=5e-4, verbose=False): 
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
        if len(Y.shape) == 1: Y = np.atleast_2d(Y).T
        if len(X.shape) == 1: X = np.atleast_2d(X).T
        
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
        if self._nde is None: 
            # default NDE architecture: Masked Autoencoder for Distribution
            # Estimation (https://arxiv.org/abs/1502.03509) with arbitrary
            # number of hidden features and blocks. 
            nde = self.set_architecture(arch='made', 
                                        nhidden=64, 
                                        ntransform=2, 
                                        nblocks=2, 
                                        num_mixture_components=2,
                                        batch_norm=True)
        nde = self._nde

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

    def set_architecture(self, arch='made', nhidden=64, ntransform=2, nblocks=2,
                         num_mixture_components=2, batch_norm=True):
        ''' Set the architecture for the flow about to be trained.

        default NDE architecture: Masked Autoencoder for Distribution
        Estimation (https://arxiv.org/abs/1502.03509) with arbitrary number of
        hidden features and blocks.
        '''
        self._nde = Ut.posterior_nn(arch, 
                    hidden_features=nhidden, 
                    num_transforms=ntransform, 
                    num_blocks=nblocks, 
                    num_mixture_components=num_mixture_components,
                    use_batch_norm=batch_norm)
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


class CausalFlowA(BaseCausalFlow): 
    def __init__(self, device=None): 
        ''' CausalFlow for Scenario A, where you're using flows to estimate
        both $p( Y | X, T=1 )$ and $p( Y | X, T=0 )$. See package README.md for
        details. 

        explain how to use it 


        Notes: 
        * Either train the treated/control flows or load them. 
        '''
        self.flow_treated = None # flow estimating p( Y | X, T=1 )
        self.flow_control = None # flow estimating p( Y | X, T=0 )

        self.support_treated = None # flow estimating p(X | T=1) 
        self.support_control = None # flow estimating p(X | T=0)
        
        super().__init__(device=device)
    
    def CATE(self, X, Nsample=10000, Nsupport=10000, support_threshold=0.95, 
             progress_bar=False, transf=None): 
        ''' evaluate the conditional average treatment effect for a given
        covariate value, X


        CATE = int Y p(Y|X,T=1) dY - int Y p(Y|X,T=0) dY

        args: 
            X: 1D array with D_x dimensions. The given covariate value to
                estimate the CATE. 
        
        kwargs: 
            Nsample: int, the number of samples used for the Monte Carlo
                integration estimates

            Nsupport: int, number of samples used to check support. 

            transf: function that specifies any transformation you want to
                perform on the flow outputs before calculating the CATE. This
                is if, e.g., you take train the flows on log Y. 
        '''
        if self.support_treated is None or self.support_control is None: 
            warnings.warn("No support flows specified! Cannot check support!")

        # check support before evaluating CATE 
        support_t = self.support_treated.check_support(X, Nsample=Nsupport,
                                           threshold=support_threshold)[0] 
        support_c = self.support_control.check_support(X, Nsample=Nsupport,
                                           threshold=support_threshold)[0]
        if not support_t:
            warnings.warn("covariate is out of the treated sample support!")
        if not support_c:
            warnings.warn("covariate is out of the control sample support!")

        # sample p( Y | X, T=1 ) 
        Ys_treated = self._sample(self.flow_treated, X, Nsample=Nsample,
                                  progress_bar=progress_bar)
        # sample p( Y | X, T=0 ) 
        Ys_control = self._sample(self.flow_control, X, Nsample=Nsample,
                                  progress_bar=progress_bar)
        if transf is not None: 
            Ys_treated = transf(Ys_treated) 
            Ys_control = transf(Ys_control) 

        # estimate the cate 
        cate = np.mean(Ys_treated) - np.mean(Ys_control) 
        return cate
    
    def train_flow_treated(self, Y_treat, X_treat, **kwargs):
        ''' Train a flow for the treated sample to estimate p( Y | X, T=1 ).
        This is a wrapper for self._train_flow. See self._train_flow for
        documentation and detail on training. 
        '''
        if self.flow_treated is not None: 
            warnings.warn("Overwriting existing flow_treated. clt+c if you don't want to do this")

        self.flow_treated = self._train_flow(Y_treat, X_treat, **kwargs) 
        return None  
    
    def train_flow_control(self, Y_cont, X_cont, **kwargs):
        ''' Train a flow for the treated sample to estimate p( Y | X, T=0 ). 
        This is a wrapper for self._train_flow. See self._train_flow for
        documentation and detail on training. 
        '''
        if self.flow_control is not None: 
            warnings.warn("Overwriting existing flow_treated. clt+c if you don't want to do this")

        self.flow_control = self._train_flow(Y_cont, X_cont, **kwargs) 
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

    def train_support_treated(self, X_treated, nhidden=128, nblock=5, batch_size=50,
                      learning_rate=5e-4, num_iter=300, clip_max_norm=5,
                      verbose=False): 
        ''' train Support.Support object for the treated sample 
        '''
        ndim = X_treated.shape[1]

        self.support_treated = Support.Support() 
        self.support_treated.set_architecture(ndim, nhidden=nhidden, nblock=nblock) 
        self.support_treated.train(X_treated, batch_size=batch_size,
                           learning_rate=learning_rate, num_iter=num_iter,
                           clip_max_norm=clip_max_norm, verbose=verbose)
        return None 
    
    def train_support_control(self, X_control, nhidden=128, nblock=5, batch_size=50,
                      learning_rate=5e-4, num_iter=300, clip_max_norm=5,
                      verbose=False): 
        ''' train Support.Support object for the control sample 
        '''
        ndim = X_control.shape[1]

        self.support_control= Support.Support() 
        self.support_contro.set_architecture(ndim, nhidden=nhidden, nblock=nblock) 
        self.support_contro.train(X_control, batch_size=batch_size,
                           learning_rate=learning_rate, num_iter=num_iter,
                           clip_max_norm=clip_max_norm, verbose=verbose)
        return None 

    def load_support_treated(self, _support): 
        ''' load support.Support object for testing support for the treated
        
        args: 
            _support: support.Support object 
        '''
        self.support_treated  = _support
        return None 
    
    def load_support_control(self, _support): 
        ''' load support.Support for testing support for the control 
        
        args: 
            _support: support.Support object 
        '''
        self.support_control  = _support
        return None 


class CausalFlowB(BaseCausalFlow): 
    def __init__(self, base, device=None): 
        ''' CausalFlow for Scenario B, where you're using flows to estimate
        only $p( Y | X, T=1 )$ or $p( Y | X, T=0 )$, but not both. This method
        is equivalent to ``synthetic matching''. 

        args: 
            base: str, specify which sample is the base distribution. Options
            are 'treated' or 'control'
        '''
        # specify which sample is the base distribution 
        if base not in ['control', 'treated']: 
            raise ValueError("specify base distribution. Options: 'control', 'treated'")
        self.base = base 

        # flow estimating p( Y | X ) of the base sample (could be either
        # treated or control 
        self.flow_base = None 

        # test support for base distribution p(X) 
        self.support_base = None 
        
        super().__init__(device=device)

    def CTE(self, X, Y, Nsample=10000, Nsupport=10000, support_threshold=0.95, 
             progress_bar=False, transf=None): 
        ''' evaluate the conditional treatment effect for a given covariate X
        and its outcome Y. Note that this is NOT the conditional AVERAGE
        treatmenet effect! See package README.md for details. 

        args: 
            X: 1D array with D_x dimensions. The given covariate value to
                estimate the CTE. 

            Y: 1D array with D_y dimensions. The given outcome to estimate the
            CTE. 
        
        kwargs: 
            Nsample: int, the number of samples used for the Monte Carlo
                integration estimates

            Nsupport: int, number of samples used to check support. 

            transf: function that specifies any transformation you want to
                perform on the flow outputs before calculating the CATE. This
                is if, e.g., you take train the flows on log Y. 
        '''
        if self.support_base is None: 
            warnings.warn("No support flows specified! Cannot check support!")

        # check support before evaluating CATE 
        support_base = self.support_base.check_support(X, Nsample=Nsupport,
                                                       threshold=support_threshold)[0]
        if not support_base:
            warnings.warn("covariate is out of the base sample support!")

        # sample p( Y | X, T=1 ) 
        Ys_base = self._sample(self.flow_base, X, Nsample=Nsample,
                               progress_bar=progress_bar)
        if transf is not None: 
            transf = lambda x: x 

        Ys_base = transf(Ys_base) 
        Y_other = transf(Y) 

        # estimate the cate 
        if self.base == 'control': 
            cte = Y_other - np.mean(Ys_base) 
        elif self.base == 'treated': 
            cte = np.mean(Ys_base) - Y_other
        return cte

    def CATE(self, Xs, Ys, Nsample=10000, Nsupport=10000, support_threshold=0.95, 
             progress_bar=False, transf=None): 
        ''' evaluate the conditional treatment effect for a given set of
        covariates Xs and their outcome Ys then average them to estimate the
        CATE. See package README.md for details. 

        args: 
            Xs: N x D_x array. The given covariates to estimate the CATE. 

            Ys: N x D_y array. The given outcomes to estimate the CATE. 
        
        kwargs: 
            Nsample: int, the number of samples used for the Monte Carlo
                integration estimates

            Nsupport: int, number of samples used to check support. 

            transf: function that specifies any transformation you want to
                perform on the flow outputs before calculating the CATE. This
                is if, e.g., you take train the flows on log Y. 
        '''

        if self.support_base is None: 
            warnings.warn("No support flows specified! Cannot check support!")

        # check support before evaluating CATE 
        support_base = self.support_base.check_support(Xs, Nsample=Nsupport,
                                                       threshold=support_threshold)[0]
        if not np.any(support_base):
            warnings.warn("some covariate is out of the base sample support!")

        # sample p( Y | X, T=1 ) 
        Ys_base = []
        for X in Xs: 
            _Y = self._sample(self.flow_base, X, Nsample=Nsample,
                                   progress_bar=progress_bar)
            Ys_base.append(_Y)
        Ys_base = np.array(Ys_base)

        if transf is None: transf = lambda x: x 

        Ys_base = transf(Ys_base) 
        Ys_other = transf(Ys) 

        # estimate the cate 
        if self.base == 'control': 
            cate = np.mean(Ys_other - Ys_base) 
        elif self.base == 'treated': 
            cate = np.mean(Ys_base - Ys_other) 
        return cate

    def train_flow(self, Y_base, X_base, **kwargs):
        ''' Train a flow for the base sample to estimate p( Y | X ).  This is a
        wrapper for self._train_flow. See self._train_flow for documentation
        and detail on training. 
        '''
        if self.flow_base is not None: 
            warnings.warn("Overwriting existing flow_base. clt+c if you don't want to do this")

        self.flow_base = self._train_flow(Y_base, X_base, **kwargs) 
        return None  

    def load_flow(self, _dir, n_ensemble=5, flow_name=None): 
        ''' load flow that estimates p( Y | X ) for the base sample, given
        either a directory or filename of the flow. By default it will load an
        ensemble of flows based on an optuna study. 

        args: 
            _dir: string specifying the directory path of the control flows  

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 
            flow_name: (str) specify the filename of the flow. If `flow_name`
                is specified, it will read that flow. 
        '''
        self.flow_base = self._load_flow(_dir, n_ensemble=5, flow_name=None)
        return None 

    def train_support(self, X_base, nhidden=128, nblock=5, batch_size=50,
                      learning_rate=5e-4, num_iter=300, clip_max_norm=5,
                      verbose=False): 
        ''' train Support.Support object. 
        '''
        ndim = X_base.shape[1]

        self.support_base = Support.Support() 
        self.support_base.set_architecture(ndim, nhidden=nhidden, nblock=nblock) 
        self.support_base.train(X_base, batch_size=batch_size,
                           learning_rate=learning_rate, num_iter=num_iter,
                           clip_max_norm=clip_max_norm, verbose=verbose)
        return None 

    def load_support(self, _support): 
        ''' load support.Support object for testing support for the base
        sample.
        
        args: 
            _support: support.Support object 
        '''
        self.support_base  = _support
        return None 
    

class CausalFlowC(BaseCausalFlow): 
    def __init__(self, device=None): 
        ''' CausalFlow for Scenario C, where you're using a flow to estimate
        $p( Y | X, T )$. This is for the scenario where treatment is not
        binary but continuous.  


        The treatment effect is estimated by comparing 
        $p( Y | X_fid, T_fid + dT )$ and $p( Y | X_fid, T_fid )$. 
        X_fid a fiducial value of the covariates. T_fid is the fiducial
        treatment level, while T' is T_fid + dT where dT is the ``treatment''.
        '''
        # flow estimating p( Y | X, T ) of the base sample (could be either
        # treated or control 
        self.flow = None 

        # test support for base distribution p(X) 
        self.support = None 
        
        super().__init__(device=device)

    def CATE(self, X_fid, T_fid, dT, Nsample=10000, Nsupport=10000, support_threshold=0.95, 
             progress_bar=False, transf=None): 
        ''' evaluate the conditional treatment effect for a given covariate X
        and its outcome Y. Note that this is NOT the conditional AVERAGE
        treatmenet effect! See package README.md for details. 

        args: 
            X_fid: 1D array with D_x dimensions. Fiducial covariate 

            T_fid: 1D array with D_t dimensions. Fiducial treatment 

            dT: 1D array with D_t dimensions. "treatment" 
        
        kwargs: 
            Nsample: int, the number of samples used for the Monte Carlo
                integration estimates

            Nsupport: int, number of samples used to check support. 

            transf: function that specifies any transformation you want to
                perform on the flow outputs before calculating the CATE. This
                is if, e.g., you take train the flows on log Y. 
        '''
        if self.support is None: 
            warnings.warn("No support flows specified! Cannot check support!")

        # check support before evaluating CATE 
        XT_fid = np.concatenate([X_fid, T_fid]) 
        supported = self.support.check_support(XT_fid, Nsample=Nsupport,
                                             threshold=support_threshold)[0]
        if not supported: warnings.warn("(Xfid, Tfid) is out of support!")
        
        # check support before evaluating CATE 
        XT_dfid = np.concatenate([X_fid, T_fid + dT]) 
        supported = self.support.check_support(XT_dfid, Nsample=Nsupport,
                                             threshold=support_threshold)[0]
        if not supported: warnings.warn("(Xfid, Tfid + dT) is out of support!")

        # sample p( Y | X_fid, T_fid ) 
        Ys_fid = self._sample(self.flow, XT_fid, Nsample=Nsample,
                              progress_bar=progress_bar)
        # sample p( Y | X_fid, T_fid + dT ) 
        Ys_dfid = self._sample(self.flow, XT_dfid, Nsample=Nsample,
                              progress_bar=progress_bar)

        if transf is None: transf = lambda x: x 
        Ys_fid = transf(Ys_fid) 
        Ys_dfid = transf(Ys_dfid) 

        # estimate the cate 
        cate = np.mean(Ys_dfid) - np.mean(Ys_fid) 
        return cate

    def train_flow(self, Y, X, T, **kwargs):
        ''' Train a flow for the base sample to estimate p( Y | X ).  This is a
        wrapper for self._train_flow. See self._train_flow for documentation
        and detail on training. 
        '''
        if self.flow is not None: 
            warnings.warn("Overwriting existing flow. clt+c if you don't want to do this")

        XT = np.concatenate([X, T], axis=1) 
        self.flow = self._train_flow(Y, XT, **kwargs) 
        return None  

    def load_flow(self, _dir, n_ensemble=5, flow_name=None): 
        ''' load flow that estimates p( Y | X ) for the base sample, given
        either a directory or filename of the flow. By default it will load an
        ensemble of flows based on an optuna study. 

        args: 
            _dir: string specifying the directory path of the control flows  

        kwargs: 
            n_ensemble: (int) specifying the number of flows in the ensemble. 
            flow_name: (str) specify the filename of the flow. If `flow_name`
                is specified, it will read that flow. 
        '''
        raise NotImplementedError

    def train_support(self, X, T, nhidden=128, nblock=5, batch_size=50,
                      learning_rate=5e-4, num_iter=300, clip_max_norm=5,
                      verbose=False): 
        ''' train Support.Support object. 
        '''
        XT = np.concatenate([X, T], axis=1) 
        ndim = XT.shape[1]

        self.support = Support.Support() 
        self.support.set_architecture(ndim, nhidden=nhidden, nblock=nblock) 
        self.support.train(XT, batch_size=batch_size,
                           learning_rate=learning_rate, num_iter=num_iter,
                           clip_max_norm=clip_max_norm, verbose=verbose)
        return None 

    def load_support(self, _support): 
        ''' load support.Support object for testing support for the base
        sample.
        
        args: 
            _support: support.Support object 
        '''
        self.support  = _support
        return None 
