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
# data reading part
def		raw_filenames(_sbj, _run):
	print_fname(f"{GRE}[raw_filenames]")
	raw_fnames = []
	for sbj in _sbj:
		raw_fname = eegbci.load_data(sbj, _run)
		raw_fnames.extend(raw_fname)
	return raw_fnames