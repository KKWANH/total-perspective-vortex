import time
import numpy as np
import os
import mne
from random import seed ,randint

import PyQt5

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils.ft_csp import FT_CSP  # use my own CSP
from mne.decoding import CSP  # use mne CSP

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump, load

import copy as cp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer

from utils.ft_utils import raw_filenames, fetch_data, prepare_data, \
                     filter_data, fetch_events, \
                     my_custom_loss_func
from ft_fit import ft_fit


def ft_predict(SUBJECTS, RUNS, tmin=-0.2, tmax=0.5, PREDICT_MODEL="final_model.joblib"):
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    start = time.perf_counter()

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames(SUBJECTS, RUNS), runs=RUNS)))
    labels, epochs = fetch_events(raw, tmin=tmin, tmax=tmax)
    # labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print(f"X.shape= {epochs.shape}, y.shape={labels.shape}")

    score = make_scorer(my_custom_loss_func, greater_is_better=False)

    scores = []
    predicts = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print(f"event={n:02d}, predict={pred}, label={labels[n:n + 1]}")
        scores.append(pred[0] == labels[n:n + 1][0])
        predicts.append(pred[0])

    end = time.perf_counter()
    exectime = end - start

    unit = 's'
    if exectime < 0.001:
        exectime *= 1000
        unit = 'ms'

    print('='*42)

    print(f"=     (clf.predict Mean-Accuracy={np.mean(scores):.3f} )     =")
    print(f"=     (clf.predict Mean-Accuracy={score(clf, epochs, labels):.3f} )     =")
    print(f"=     (clf.predict Exec-Time    ={exectime:.3f}{unit})     =")
    print('='*42)

    return predicts, np.mean(scores)


if __name__ == "__main__":
    RUNS1 = [3, 7, 11]  # motor: left hand vs right hand
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS3 = [5, 9, 13]  # motor: hands vs feet
    RUNS4 = [6, 10, 14] # motor imagery: hands vs feet
    RUNS = RUNS2
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)

    # Define SUBJECTS as all subjects in the dataset
    SUBJECTS = list(range(1, 110)) # assuming there are 109 subjects numbered 1 through 109

    # Initialize an empty list for overall scores
    overall_scores = []

    # Loop over all subjects
    # for SUBJECT in SUBJECTS:
        # Fit and transform the data
    ft_fit(SUBJECTS, RUNS, _min=tmin, _max=tmax)

    # Load the prediction model
    PREDICT_MODEL = "final_model.joblib"

    # Predict and get the score
    predict_, score_ = ft_predict(SUBJECTS, RUNS, tmin=tmin, tmax=tmax, PREDICT_MODEL=PREDICT_MODEL)

    # Add the score to overall scores
    overall_scores.append(score_)
    print(f"Score: {score_}")

    # Calculate and print the average score over all subjects
    avg_score = round(np.mean(overall_scores), 2)
    print(f"Average score over all subjects: {avg_score}")