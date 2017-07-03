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
import random
import numpy as np
import pandas as pd

#-----------------------------
# Import Global Variables
#-----------------------------
from acs_config import data_path, data_dc, url_dc, download, logger, train_sample, test_sample, feature_ranking_sample_dc, FeatsSelectorModel, \
                        n_feats_prel, c_feats_prel, label_id, xgb_params, output_header_dc

#-----------------------------
# User defined modules and functions
#-----------------------------
import utils
from acs_modules import DataDownloader, DataReader, FeatsManipulator, PrelFeatsSelector, Regressor, OutputHandler

#******************************************************************************
# MAIN PROGRAM
#******************************************************************************

if __name__=='__main__':

    if download:
        DataDownloader().download_from_kaggle(url_dc)
        logger.info('Data downloaded from Kaggle.')
    else:
        logger.info('Download from Kaggle skipped. Using data stored.')
    #end

    df_dc = {}
    df_dc['train_df'] = DataReader().read_train(data_dc, train_sample=train_sample)
    logger.info('Train data successfully read. Sample: ' + str(train_sample))
    df_dc['test_df'] = DataReader().read_test(data_dc, test_sample=test_sample)
    logger.info('Test data successfully read. Sample: ' + str(test_sample))

    full_df = FeatsManipulator().preliminary_manipulation(df_dc, categorical=True)
    del df_dc
    logger.info('Preliminary DataFrame manipulation successful.')

    PrelFeatSelector = PrelFeatsSelector(FeatsSelectorModel, num_threshold=n_feats_prel, cat_threshold=c_feats_prel, sample_dc=feature_ranking_sample_dc, best=True)

    feats = [col for col in full_df if col not in ['id', 'is_test']]

    feats_ranked = PrelFeatSelector.select_feats(full_df[feats][full_df['is_test']==0], label_id=label_id, feat_type='numeric')

    optimal_df = full_df[feats_ranked + ['id', 'is_test', 'loss']]

    train_df = optimal_df[optimal_df['is_test']==0].drop(['id', 'is_test', 'loss'], axis=1)
    test_df = optimal_df[optimal_df['is_test']==1].drop(['id', 'is_test', 'loss'], axis=1)
    loss_sr = optimal_df['loss'][optimal_df['is_test']==0]
    test_id_sr = optimal_df['id'][optimal_df['is_test']==1]

    cv_dc = Regressor().kfolds_cv(train_df, loss_sr, test_df, xgb_params, n_folds=5)
    xgb_params['addtn_par'] = {'num_boost_round': cv_dc['avg_n_rounds'], 'early_stopping_rounds': 25}

    pred_loss_ls = Regressor().regress(train_df, loss_sr, test_df, xgb_params)

    np.save('./pred_loss', pred_loss_ls)

    OutputHandler().output_writer(test_id_sr.tolist(), cv_dc['avg_pred_ls'], header_dc = output_header_dc, name='cv_avg_subm', gz=True)
    OutputHandler().output_writer(test_id_sr.tolist(), pred_loss_ls, header_dc = output_header_dc, name='full_train_subm', gz=True)


#end


