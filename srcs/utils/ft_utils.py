#-------------------------------------------------------------------------------
# - library
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
# pre-defined prints
def		print_fname(_str):
	print(f"{BOL}{BLU}[function excuted]{RES}")
	print(f"    : {BOL}{UND}{_str}{RES}")

#-------------------------------------------------------------------------------
# data filtering
def		filter_data(_raw):
	print_fname(f"{GRE}[filter_name]")

	data_filter = _raw.copy()
	data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
	"""
	Why 7~30Hz? Why not 8~40Hz?
	
	"""
	return data_filter

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
def		fetch_data(_raw_fnames, _runs, _sfreq, _bool_print):
	"""
Reading EEG data from given file names
	"""
	return 0

#-------------------------------------------------------------------------------
def		raw_filenames(_subjects, _runs):
	print_fname(f"{GRE}[raw_filenames]")
	raw_fnames = []
	for subject in _subjects:
		raw_fname = eegbci.load_data(subject, _runs)
		raw_fnames.extend(raw_fname)
	return raw_fnames

#-------------------------------------------------------------------------------
def		fetch_events(_data_filtered, _min=0.0, _max=2.0):
	"""
fetch evetns from data
returns label and epoch; EEG data over a continuous period of time.
	"""
	return 0, 0