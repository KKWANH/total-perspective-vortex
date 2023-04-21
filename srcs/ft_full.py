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
from	ft_color 						import	*

from	ft_csp							import	FT_CSP
from	mne.decoding					import	CSP

#-------------------------------------------------------------------------------
def		fetch_events(_data_filtered, _min=0.0, _max=2.0):
	"""
fetch evetns from data
returns label and epoch; EEG data over a continuous period of time.
	"""
	return 0, 0

#-------------------------------------------------------------------------------
def		filter_data(_raw):
	"""
filters the EEG data with 8~30Hz bandwith
	"""
	return 0

#-------------------------------------------------------------------------------
def		prepare_data(_raw, _montage):
	"""
pre-processing for EEG data.
normalize the channel name, standardize the data, set montage on data.

MONTAGE:
	a method to solve and process EEG dataset
	defines the positions of electric nodes and channel batch.
	WHY?
		standadize
		pre-processing and analysis
		location information
	the most use system of Montage is 10-20 system.
	"""
	return 0

#-------------------------------------------------------------------------------
def		fetch_data(_raw_fnames, _sfreq, _bool_print):
	"""
Reading EEG data from given file names
	"""
	return 0

#-------------------------------------------------------------------------------
def		raw_filenames():
	"""
return the raw EEG data file name
	"""
	return 0

#-------------------------------------------------------------------------------
def		ft_fit():
	"""
saves the classifier.
training classifier with CSP(Common Spatial Pattern), LDA(Linear Discriminant Analysis)
saves the trained model as "final_model.joblib"
	"""
	return 0

#-------------------------------------------------------------------------------
def		ft_predict():
	"""
loads trained model, make a predict for a new data.
	"""
	return 0

#-------------------------------------------------------------------------------
def		ft_pipeline():
	"""
executes the whole machine learning pipeline.
	""" 
	return 0

#-------------------------------------------------------------------------------
if		__name__ == "__main__":
	SBJ = [42]
	RHF = [ 6, 10, 14]	# (runs) motor imagery : hands vs feet
	RLR = [ 4,  8, 12]	# (runs) motor imagery : left hand vs right hand
	RUN = RLR

	ft_pipeline()