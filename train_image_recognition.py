# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:59:36 2017

@author: Matt Green
"""

import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))