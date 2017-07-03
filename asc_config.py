"""
  **************************************
  Created by Romano Foti - rfoti
  On 11/14/2016
  **************************************
"""
#******************************************************************************
# Importing packages
#******************************************************************************
#-----------------------------
# Standard libraries
#-----------------------------
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import make_scorer
#-----------------------------
# User defined modules and functions
#-----------------------------
import utils

#******************************************************************************

#******************************************************************************
# MAIN PROGRAM
#******************************************************************************

data_path = './'
header = True

data_dc = {'train.csv': {'args':{'dtype': {'id': np.int32}}, 'train': True, 'header': header, 'data_path': data_path},
           'test.csv': {'args':{'dtype': {'id': np.int32}}, 'train': False, 'header': header, 'data_path': data_path},
            }

url_dc = {
          'train': 'https://www.kaggle.com/c/allstate-claim-severity/download/train.csv.zip',
          'test': 'https://www.kaggle.com/c/allstate-claim-severity/download/test.csv.zip',
          'sample_subm': 'https://www.kaggle.com/c/allstate-claim-severity/download/sample_submission.csv.zip'
          }

download = True

train_sample = None #None if no sampling is required
test_sample = None #None if no sampling is required

feature_ranking_sample_dc = {'numeric': 0.5, 'categorical': 0.25}

Feats_Selector_Classifier = RFC(n_estimators=20)
n_feats_prel = 800
c_feats_prel = 20

label_id = 'Response'
score = make_scorer(MCC, greater_is_better=True)

xgb_params = {'params': {
                         'seed': 0,
                         'colsample_bytree': 0.8,
                         'silent': 1,
                         'subsample': 0.8,
                         'learning_rate': 0.01,
                         'num_boost_round': 100,
                         'max_depth': 7,
                         'num_parallel_tree': 1,
                         'min_child_weight': 2,
                         'scale_pos_weight': 1,
                         'feval': score,
                         'base_score': 0.05
                         },
               'cross_val': {'num_boost_round': 10, 
                           'nfold': 5, 
                           'seed': 0, 
                           'stratified': True,
                           'early_stopping_rounds': 1,
                           'verbose_eval': 1,
                           'show_stdv': True
                           }
              }

logger = utils.Logging().configure_logger('bpsp_model_logs', './bpsp_logfile.log')

#******************************************************************************
