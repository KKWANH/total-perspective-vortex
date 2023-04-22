#-------------------------------------------------------------------------------
import 	copy							as cpy
import	numpy							as npy
import	matplotlib.pyplot				as plt
from	sklearn.base					import	(
        BaseEstimator,
        TransformerMixin)

#-------------------------------------------------------------------------------
class	FT_CSP(TransformerMixin, BaseEstimator):
    """
    """
    def	__init__(
        self,
        n_components = 4,
        reg = None,
        log = None,
        cov_est = 'concat',
        transform_info = 'average_power',
        norm_trace = False,
        cov_method_params = None,
        rank = None,
        component_order = 'mutual_info'
    ):
        self.bool_print = False
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        self.reg = reg
        self.log = log
        
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError('unknown covariance estimation method')
        
        self.cov_est = cov_est
        self.transform_into = transform_info
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params
        self.rank = rank
        self.n_component_order = component_order

    def _check_Xy(
        self,
        X,
        y=None
    ):
        """
        """
        if not isinstance(X, npy.ndarray):
            raise ValueError(f"X should be of type ndarray (got {type(X)})")