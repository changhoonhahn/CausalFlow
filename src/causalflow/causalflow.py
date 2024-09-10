'''


main causalflow module 


'''
import numpy as np 


class BaseCausalFlow(object): 
    def __init__(self): 
        ''' Base CausalFlow class 
        '''

    def _train_flow(self): 
        ''' function for training flows 
        '''


class CausalFlowA(BaseCausalFlow): 
    def __init__(self, ): 
        ''' CausalFlow for Scenario A, where you're using flows to estimate
        both $p( Y | X, T=1 )$ and $p( Y | X, T=0 )$
        '''



class CausalFlowB(BaseCausalFlow): 
    def __init__(self, ): 
        ''' CausalFlow for Scenario B, where you're using flows to estimate
        only $p( Y | X, T=1 )$ or $p( Y | X, T=0 )$, but not both. This method
        is equivalent to ``synthetic matching''. 
        '''


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

