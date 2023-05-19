#-------------------------------------------------------------------------------
import 	copy							as cpy
import	numpy							as npy
import	matplotlib.pyplot				as plt
from	sklearn.base					import	(
        BaseEstimator,
        TransformerMixin)
from    ft_color                        import  *
from    ft_utils                        import  print_fname

#-------------------------------------------------------------------------------
"""

WHAT IS CSP?
    CSP(Common Spatial Pattern)은 공간 필터링 기법입니다.
    뇌파 데이터를 분류하는 알고리즘이며, 두 클래스 간의 분리를 최대화하는 공간 필터를 찾는 것이 목표입니다.
    WHAT IS SPATIAL FILTER?
        원래의 고차원 뎅터를 새로운 표현 공간으로 변환하는 것입니다!
    "두 클래스 간의 분리"란?
        각 클래스의 특성을 구분하는 정도를 나타내며, 이 값이 높을 수록 두 클래스는 비교하기 쉽습니다.
        두 클래스가 가장 잘 구별되는 특성 공간을 찾는 게 목표입니다.

"""

#-------------------------------------------------------------------------------
class	FT_CSP(TransformerMixin, BaseEstimator):
    #---------------------------------------------------------------------------
    # setup
    def __init__(
        self,
        n_components        = 4,                # 생성된 특성 공간의 차원 수 결정.
                                                # 정수
                                                # 고차원 데이터를 n차원 특성 공간으로 축소함.
                                                # 값이 높아지면 더 많은 차원의 특성 공간으로 데이터를 변환함.
                                                # 더 많은 고유벡터를 고려할 수 있지만 값이 너무 크면 노이즈를 포함하는 차원이 결과에 영향을 줘서 성능 저하..
                                                # 너무 낮은 차원으로 축소하면 중요한 정보가 손실되어 분류 성능 저하 ..
        reg                 = None,             # 공분산 행렬에 정규화를 적용하는데 사용됨
                                                # None/실수
                                                # 공분산 행렬에 노이즈가 있거나 계산이 불안정한 경우 안정성을 높이기 위해 사용
                                                # 작은 양수 값(0.1, 0.01, 0.001)을 사용하는 게 일반적임.
                                                # 공분산 행렬의 대각 성분에 추가해서 행렬이 역행렬을 갖도록 함.
                                                # 이렇게 하면 데이터에 있는 노이즈와 이상치에 덜 민감한 결과가 입력됨
                                                # 높을 수록 노이즈에 덜 민감하게 되지만 일부 정보가 손실 될 수 있음.
                                                # 낮을 수록 이상치에 더 민감한 결과를 얻게 됨.
        log                 = None,             # 로그 변환 여부
        cov_est             = 'concat',         # 공분산 추정 방법
                                                # concat / epoch
                                                # - concat:
                                                #   모든 epoch의 데이터를 연결(concatenate)한 후 연결된 데이터를 기반으로 공분산 행렬을 계산
                                                #   시간 구간이 겹치지 않은 데이터를 다루는데 적합
                                                #   데이터를 긴 하나의 시계열 데이터로 간주하고 전ㄴ체 범위에 대한 공분산 행렬을 얻을 수 있음.
                                                # - epoch:
                                                #   각 epoch에 대한 공분산 행렬을 개별적으로 계산한 후 평균화해서 최종 공분산 행렬을 얻음
                                                #   각 epoch가 독립적인 정보를 가지고 있으며 epoch 간 변동이 중요한 경우 적합
        transform_into      = 'average_power',  # 변환 옵션
        norm_trace          = False,            # 공분산 행렬 추적 정규화 여부
        cov_method_params   = None,             # 공분산 메서드
        rank                = None,             # 랭크
        component_order     = 'mutual_info'     # CSP 컴포넌트 순서
    ):
        """
        default setting
        """
        if  not isinstance(n_components, int):
            raise ValueError(f"Argument {GRE}{BOL}{UND}[n_component]{RES} must be an integer type.")
        if  not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError(f"{RED}{BOL}Unknown covariance estimation method!{res} Argument {GRE}{BOL}{UND}[cov_est]{RES} must be \"concat\" or \"epoch\".")
        
        self.PRT                = False
        self.n_component        = n_component
        self.reg                = reg
        self.log                = log
        self.cov_est            = cov_est
        self.transform_into     = transform_into
        self.norm_trace         = norm_trace
        self.cov_method_params  = cov_method_params
        self.rank               = rank
        self.component_order    = component_order

    #---------------------------------------------------------------------------
    # sub functions
    def _check_input_shapes(
        self,
        X,
        y   = None
    ):
        """
        check input variable(X, y)'s shape.
        """
        if  not isinstance(X, npy.ndarray):
            raise ValueError(f"Variable {MAG}{BOL}{UND}[X]{RES} must be type of ndarray, but got {type(X)}")
        if  y is not None and (len(X) != len(y) or len(y) < 1):
            raise ValueError(f"Arguments {MAG}{BOL}{UND}[X, y]{RES} must have same length, but got: [X:{len(X)}] [y:{len(y)}]")
        if  X.ndim < 3:
            raise ValueError(f"Variable {MAG}{BOL}{UND}[X]{RES} must have at lest 3 dimensions.")
    
    def _compute_covariance_matrices(
        self,
        X,
        y
    ):
        _, n_channels, _ = X.shape

        covs = []
        sample_weights = []

        for _class in self._classes:
            cov, sample_weight = self._concat_cov(X[y == _class])
            if  self.norm_trace:
                cov /= npy.trace(cov)
            
            covs.append(cov)
            sample_weights.append(sample_weight)
        
        if  self.PRT:
            print(f"    - ft_CSP/_compute_covariance_matrices")
            print(f"        : weights : {weights}")

        return  npy.stack(covs), npy.array(sample_weights)
    
    def _decompose_covs(
        self,
        covs,
        sample_weights
    ):
        from scipy import linalg
        n_classes = len(covs)
        if  n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise NotImplementedError(f"Sorry, decomposing covs for the case of more than 2 classes is not implemented.")
        return  eigen_vectors, eigen_values

    def _pinv2(
        a,
        rtol=1e-05
    ):
        """
        Computing a pseudo-inverse of a matrix.
        """
        u, s, vh = npy.linalg.svd(a, full_matrices=False)
        del a
        maxS = npy.max(s)
        if  rtol is None:
            rtol = max(vh.shape + u.shape) * npy.finfo(u.dtype).eps
        rank = npy.sum(s > maxS * rtol)
        u = u[:, :rank]
        u /= s[:rank]
        return (u @ vh[:rank]).conj().T

    def _order_components(
        self,
        covs,
        sample_weight,
        eigen_vectors,
        eigen_values
    ):
        n_classes = len(self._classes)
        if n_classes > 2:
            raise NotImplementedError(f"Sorry, case of more than two classes is not implemented.")
        elif n_classes == 2:
            ix = npy.argsort(npy.abs(eigen_values - 0.5))[::-1]
        return  ix
    
    #---------------------------------------------------------------------------
    # fit
    def fit(
        self,
        X,      # ndarray
                # shape (n_epochs, n_channels, n_times)
                # data on which to estimate the CSP
        y       # array
                # shape (n_epochs, )
                # class for each epoch
    ):
        if  self.PRT:
            print_fname(f"{YEL}ft_CSP/fit")
            print(f"    : csp.fit(X, y), X.shape={X.shape}, y.shape={y.shape}")
        self._check_input_shapes(X, y)
        if  self.PRT:
            print(f"    : [lenX:{len(X)}] [lenY:{len(y)}]")
            print(f"    : y : {y}")
        
        self._classes = npy.unique(y)
        if  self.PRT:
            print(f"    : _classes : {self._classes}")
        n_classes = len(self._classes)
        if  n_classes < 2:
            return  ValueError("Variable {CYA}{BOL}{UND}[n_classes]{RES} must be bigger than 2.(>= 2)")
        
        # Computing Covariance Matrix of Each Class
        # S1 = npy.cov(X1), S2 = npy.cov(X2) of csp.py
        covs, sample_weights = self._compute_covariance_matrices(X, y)
        if  self.PRT:
            print(f"    : covs.shape : {covs.shape}")
            print(f"    : sample_weights : {sample_weights}")
        
        # Solve the eigenvalue problem S1 * W = l * S2 * W
        eigen_vectors, eigen_values = self._decompose_covs(covs, sample_weights)
        if  self.PRT:
            print(f"    : eigen_vectors.shape : {eigen_vectors.shape}")
            print(f"    : eigen_values.shape : {eigen_values.shape}")
        if	self.PRT:
            print(f"    : npy.abs(eigen_values - 0.5) = {npy.abs(eigen_values - 0.5)}\n")
            print(f"    : npy.argsort(npy.abs(eigen_values - 0.5)) = {npy.argsort(npy.abs(eigen_values - 0.5))}\n")
            print(f"    : npy.argsort(npy.abs(eigen_values - 0.5))[::-1] = {npy.argsort(npy.abs(eigen_values - 0.5))[::-1]}\n")
        
        # 64 dimensions to n_component = 4
        #	If 'mutual_info' order components by decreasing mutual information
        # 	(in the two-class case this uses a simplification)
        #	which orders components by decreasing absolute deviation of the eigenvalues from 0.5
        #	https://hal.science/hal-00602686/document
        #	ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        ix = self._order_components(covs, sample_weights, eigen_vectors, eigen_values)
        if  self.PRT:
            print(f"    : ix = {ix}")
        
        self.filters_   = eigen_vectors.T
        self.patterns_  = self._pinv2(eigen_vectors)
        pick_filters    = self.filters_[:self.n_component]

        X           = npy.asarray.dot([npy.dot(pick_filters, epoch) for epoch in X])
        X           = (X ** 2).mean(axis=2)
        self.mean_  = X.mean(axis=0)
        self.std_   = X.std(axis=0)
        
        return  self
    
    #---------------------------------------------------------------------------
    # transform
    def transform(
        self,
        X       # array, shpae(n_epochs, n_channels, n_times)
                # The data.
    ):
        """
        Estimate epochs sources given the CSP filters.
        Returns:
            X:  ndarray
                if self.transform_into == 'average_power' then returns the power of
                CSO features averaged over time and shape(n_epochs, n_sources)
        """
        if  self.PRT:
            print_fname(f"{YEL}ft_CSP/transform")
        
        if  not isinstance(X, npy.ndarray):
            raise ValueError(f"Sorry. X must be of type ndarray, but got {type(X)}.")
        if  self.filters_ is None:
            raise RuntimeError(f"No filters available. Please fit CSP before transform(decomposition)")
        
        pick_filters = self.filters_[:self.n_component]

        X = npy.asarray([npy.dot(pick_filters, epoch) for epoch in X])

        if  self.transform_into == 'average_power':
            X   = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if  log:
                X =  npy.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        
        return  X
    
    #---------------------------------------------------------------------------
    # fit_transform
    def fit_transform(
        self,
        X,              # array, shape(n_samples, n_features)
                        # Training set
        y,              # array, shape(n_samples)
                        # Traget values or class labels
        **fit_params    # dict
                        # Additional fitting parameters passed to the ``fit`` method.
    ):
        """
        Fit to data, then transform it.
        Fits transformer to ``X`` and ``u`` with optional parameters
        ``fit_params``, and returns a transformed version of ``X``.
        Return:
            X_new:  array, shape(n_samples, n_features_new)
                    Transformed array
        """
        self.fit(X, y)
        return  self.transform(X)
    
    #---------------------------------------------------------------------------
    def plot_patterns(
        self,
        info,                           # informations about data
        components      = None,         # 시각화할 컴포넌트의 인덱스/배열 
        *,                              # !!이 뒤에 오는 인자들은 키워드 전용 인자로 간주됨
        average         = None,         # 평균을 계산할 방법을 지정
        ch_type         = None,         # 채널 유형 지정
        scalings        = None,         # 채널 스케일링 지정
        sensors         = True,         # 센서 표시할지 여부
        show_names      = False,        # 채널 이름 표시할지 여부
        mask            = None,         # 마스크 적용할 때 사용
        mask_parms      = None,         # 마스크 파라미터 지정
        contours        = 6,            # 등고선의 수
        outlines        = 'head',       # 플롯 외각선의 종류 지정
        sphere          = None,         # 플롯의 구 형상 지정
        image_interp    = 'cubic',      # 이미지 보간 방법 지정
        extrapolate     = 'auto',       # 측정값 없는 영역의 외삽 방법 지정
        border          = 'mean',       # 외곽 영역의 처리 방법을 지정
        res             = 64,           # 플롯의 해상도를 지정
        size            = 1,            # 플롯의 크기 지정
        cmap            = 'RdBu_r',     # 컬러맵 지정
        vlim            = (None, None), # 컬러바 범위 지정
        cnorm           = None,         # 컬러맵 정규화 지정
        colorbar        = True,         # 컬러바 표시 여부 지정
        cbar_fmt        = '%3.1f',      # 컬라바 포맷 지정
        units           = None,         # 플룻의 단위 지정
        axes            = None,         # 사용할 축 지정
        name_format     = 'CSP%01d',    # 컴포넌트 이름의 포맷 지정
        nrows           = 1,            # 그리드 플롯의 행 수 지정
        ncols           = 'auto',       # 그리드 플롯의 열 수 지정
        show            = True          # 플룻 표시할지 여부 지정
    ):
        print_fname(f"{YEL}ft_csp/plot_patterns")

        from mne import EvokedArray

        if  units is None:
            units = 'AU'
        if  components is None:
            components = npy.arange(self.n_component)
        
        info = cpy.deepcopy(info)
        with info._unlock():
            info['streq'] = 1.
        
        patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        fig = patterns.plot_topomap(
            times           = components,
            average         = average,
            ch_type         = ch_type,
            scalings        = scalings,
            sensors         = sensors,
            show_names      = show_names,
            mask            = mask,
            mask_params     = mask_parms,
            contours        = contours,
            outlines        = outlines,
            sphere          = sphere,
            image_interp    = image_interp,
            extrapolate     = extrapolate,
            border          = border,
            res             = res,
            size            = size,
            cmap            = cmap,
            vlim            = vlim,
            cnorm           = cnorm,
            colorbar        = colorbar,
            cbar_fmt        = cbar_fmt,
            units           = units,
            axes            = axes,
            time_format     = name_format,
            nrows           = nrows,
            ncols           = ncols,
            show            = show
        )
        fig.show()