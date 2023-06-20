#-------------------------------------------------------------------------------
# library
import	numpy							as		npy
import 	os
import  matplotlib.pyplot               as      plt
import 	mne
import  PyQt5

from	mne 							import	events_from_annotations
from	mne.io 							import  (
        concatenate_raws,
        read_raw_edf)
from	mne.datasets					import	eegbci

import 	sklearn
from 	sklearn.svm						import SVC
from	sklearn.discriminant_analysis	import  (
        LinearDiscriminantAnalysis 		as      LDA)
from	sklearn.model_selection			import  (
        ShuffleSplit,
        cross_val_score)
from	sklearn.pipeline				import	Pipeline
from	sklearn.base					import	(
        BaseEstimator,
        TransformerMixin)

from	joblib							import	dump, load
import 	copy							as cpy
from	utils.ft_color 					import	*

from	utils.ft_csp					import	FT_CSP
from	mne.decoding					import	CSP

from	utils.ft_utils					import	(
        print_fname,
        filter_data,
        prepare_data,
        fetch_data,
        raw_filenames,
        fetch_events)

#-------------------------------------------------------------------------------
# setting
os.environ['MNE_DATA'] = os.path.expanduser('~/goinfre/mne_data')

#-------------------------------------------------------------------------------
# fit
def		ft_fit(
    _subjects,
    _runs,
    _min        = -0.2,
    _max        = 0.5,
    _bool_print = False,
    _bool_plot  = False,
    _bool_pipe  = True):

    print_fname(f"{CYA}[ft_fit]")
    """
saves the classifier.
training classifier with CSP(Common Spatial Pattern), LDA(Linear Discriminant Analysis)
saves the trained model as "final_model.joblib"
    """
    raw = filter_data(
        raw = prepare_data(
            raw = fetch_data(
                raw_fnames = raw_filenames(
                    subjects = _subjects,
                    runs = _runs
                ),
                runs = _runs
            )
        )
    ) # get raw data by subjects and runs

    label, epoch = fetch_events(
        data_filtered = raw,
        tmin = _min,
        tmax = _max
    ) # separate label and epoch

    epoch_train = epoch.copy().crop(
        tmin = _min,
        tmax = _max
    ) # cut epoch data by time range
    epoch_rdata = epoch.get_data()				# get raw data for full epoch
    epoch_train_rdata = epoch_train.get_data()	# get raw data for cropped epoch
    if _bool_print == True:
        print(f"    {BOL}epoch_rdata{RES} :", epoch_rdata)
        print(f"    {BOL}epoch_train_rdata{RES} :", epoch_rdata)
    
    # defining monte-carlo cross-validation generator
    """
@CONCEPT: Monte-Carlo Cross Validation
    Shuffles the data randomly and evaluates the average performance of the data.
    parameter:
        5:
            Split data for 5 times
        test_size=0.2:
            Use the 20% of data as test set.
            Rest of data(80%) will be train set.
        random_state=42:
            Seed of random generator
    """
    cross_validator = ShuffleSplit(5, test_size=0.2, random_state=42)

    # assembling classifier
    """
@CONCEPT: LDA (Linear Discriminant Analysis, 선형 판별 분석)
    one of classification algorithm on supervised learning
    GOAL:
        - find a new axis that can classify between different classes
        - project it onto a lower-dimensional space
            transforming original high-dim data onto low-dim.
            this transforming can saves feature of data and reduces dimension
             -> increase the performance
            EXAMPLE:
                데이터 포인트가 3차원(x, y, z)에서 주어진 경우,
                각 데이터 포인트는 3개의 값을 갖습니다.
                이를 저차원 공간, 예를 들어 2차원(x', y')으로 투영하면,
                각 데이터 포인트는 이제 2개의 값만을 갖게 됩니다.
            
@CONCEPT: shrinkage
    정규화의 일종(regularation)
    LDA에서 공분산 행렬을 추정할 때 사용
    overfitting 방지, 일반화 성능 향상
    훈련 샘플이 적거나 고차원 데이터인 경우 성능 향상
    """
    lda = LDA(
        solver = 'lsqr',
        shrinkage = 'auto'
    )

    """
@CONCEPT: CSP
    """
    csp = FT_CSP(
        n_components    = 4,
        reg             = None,
        log             = True,
        norm_trace      = False)

    classfication = Pipeline([('CSP', csp), ('LDA', lda)])

    if _bool_pipe is True:
        scores_lda_shrinkage        = cross_val_score(classfication, epoch_train_rdata, label)
        mean_score_lda_shrinkage    = npy.mean(scores_lda_shrinkage)
        std_scores_lda_shrinkage    = npy.std(scores_lda_shrinkage)
    else:
        epoch_train_rdata_csp       = csp.fit_transform(epoch_train_rdata, label)
        classfication2              = Pipeline([('LDA', lda)])
        scores_lda_shrinkage        = cross_val_score(classfication2, epoch_train_rdata_csp, label, cv=cross_validator, n_jobs=1)
        mean_score_lda_shrinkage    = npy.mean(scores_lda_shrinkage)
        std_scores_lda_shrinkage    = npy.std(scores_lda_shrinkage)
    
    class_balance = npy.mean(label == label[0])
    class_balance = max(class_balance, 1 - class_balance)
    print(f"")
    print(f"    LDA shrinked classification accuracy : {npy.mean(scores_lda_shrinkage)} / Chance level: {class_balance}")
    print(f"    Mean Score Model {mean_score_lda_shrinkage}")
    print(f"    Std Score Model {std_scores_lda_shrinkage}")
    print(f"")

    csp.fit_transform(
        epoch_train_rdata,
        label)
    csp.plot_patterns(
        epoch.info,
        ch_type     = 'eeg',
        units       = 'Patterns (AU)',
        size        = 1.5)

    classfication = classfication.fit(
        epoch_train_rdata,
        label)
    dump(
        classfication,
        "final_model.joblib")
    print(f"    - model saved to final_model.joblib")

    return  raw

if  __name__ == "__main__":
    RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS = RUNS2

    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)

    plt.ioff()
    SUBJECTS = [13]

    subject_list=[ 1,  3,  4,  8,  9, 12, 13, 15, 17, 18,
                  19, 20, 21, 22, 25, 26, 36, 37, 40, 41,
                  42, 46, 47, 48, 50, 51, 53, 54, 61, 62,
                  63, 68, 71, 73, 77, 80, 83, 84, 85, 86,
                  87, 90, 93, 98, 100, 101, 102, 103, 104, 105]

    raw = ft_fit(
        SUBJECTS,
        RUNS,
        _min=tmin,
        _max=tmax)

    # plt.ion()
    # fig = plt.figure(figsize=(4.2, 4.2))
    # plt.plot(range(10), range(10))
    plt.show()
