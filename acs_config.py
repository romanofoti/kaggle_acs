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
from sklearn.ensemble import RandomForestRegressor as RFR
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
          'train.csv.zip': 'https://www.kaggle.com/c/allstate-claims-severity/download/train.csv.zip',
          'test.csv.zip': 'https://www.kaggle.com/c/allstate-claims-severity/download/test.csv.zip',
          'sample_submission.csv.zip': 'https://www.kaggle.com/c/allstate-claims-severity/download/sample_submission.csv.zip'
          }

download = True

train_sample = None #None if no sampling is required
test_sample = None #None if no sampling is required

feature_ranking_sample_dc = {'numeric': 0.25, 'categorical': 0.25}

FeatsSelectorModel = RFR(n_estimators=20)
n_feats_prel = 800
c_feats_prel = 20

label_id = 'loss'

xgb_params = {'params': {
                         'booster': 'gbtree',
                         'objective': 'reg:linear',
                         'eva_metric': 'mae',
                         'seed': 0,
                         'colsample_bytree': 0.6,
                         'subsample': 0.9,
                         'learning_rate': 0.1,
                         'gamma': 0.6,
                         'num_boost_round': 100,
                         'max_depth': 7,
                         'num_parallel_tree': 1,
                         'min_child_weight': 4,
                         'max_delta_step': 0,
                         },
               'cross_val': {'num_boost_round': 10000, 
                             'nfold': 5, 
                             'seed': 0, 
                             'stratified': True,
                             'early_stopping_rounds': 25,
                             'verbose_eval': 1,
                             'show_stdv': True
                             }
              }


output_header_dc = {'id': 'id', 'label': 'loss'}

logger = utils.Logging().configure_logger('acs_model_logs', './acs_logfile.log')

#******************************************************************************
