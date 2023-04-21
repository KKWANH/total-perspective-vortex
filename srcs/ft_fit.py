#-------------------------------------------------------------------------------
# library
import	numpy							as		npy
import 	os
import 	mne
from	mne 							import	eventsfrom_annotations
from	mne.io 							import (
    	concatenate_raws,
		read_raw_eof)
from	mne.datasets					import	eegbci

import 	sklearn
from 	sklearn.svm						import SVC
from	sklearn.discriminant_analysis	import (
    	LinearDiscriminantAnalysis 		as LDA)
from	sklearn.model_selection			import (
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
def		ft_fit(_subjects, _runs, _min=-0.2, _max=0.5, _bool_print=False, _bool_plot=False, _bool_pipe=True):
	print_fname(f"{CYA}[ft_fit]")
	"""
saves the classifier.
training classifier with CSP(Common Spatial Pattern), LDA(Linear Discriminant Analysis)
saves the trained model as "final_model.joblib"
	"""
	raw = filter_data(
		_raw = prepare_data(
			_raw = fetch_data(
				_raw = raw_filenames(
					_subjects = _subjects,
					_runs = _runs
				),
				_runs = _runs
			)
		)
	) # get raw data by subjects and runs

	label, epoch = fetch_events(
		_data_filtered = raw,
		_min = _min,
		_max = _max
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
@TODO
@CONCEPT: LDA
@CONCEPT: shrinkage
	"""
	lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
	"""
@TODO
@CONCEPT: CSP
	"""
	# ft_csp = FT_CSP(n_components)

