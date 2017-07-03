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
import pickle
import numpy as np
import pandas as pd
import scipy
from datetime import datetime
from collections import defaultdict
from scipy import sparse
from scipy.stats import skew, boxcox
from math import exp, log
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, Imputer
import xgboost as xgb
from bayes_opt import BayesianOptimization

#-----------------------------
# User defined modules and functions
#-----------------------------
import kaggle_utils
import utils
from acs_config import logger

#******************************************************************************

#******************************************************************************
# Defining functions
#******************************************************************************

class DataDownloader():
    '''
    '''

    def __init__(self):
        pass
    #end

    def download_from_kaggle(self, url_dc=None):
        '''
        Downloads and unzips datasets from Kaggle

        '''
        if url_dc==None:      
            logger.info('Error: Dictionary of downloading URLs needs to be provided!')
        #end
        for ds, url in zip(url_dc.keys(), url_dc.values()):
            logger.info('Downloading and unzipping %s ...' %ds)
            kaggle_utils.KaggleRequest(logger=logger).retrieve_dataset(url)
        #end
        return
    #end

#end

class DataReader():
    '''
    '''
    def __init__(self):
        pass
    #end

    def sample_skip(self, filename, samplesize=1000, header=True):
        '''
        Reads a random sample of lines from a csv file and
        loads it into a pandas DataFrame
        '''
        num_records = sum(1 for line in open(filename)) - int(header)
        skip = sorted(random.sample(xrange(int(header), num_records+1), num_records-samplesize))
        return skip
    #end

    def read_train(self, data_dc, train_sample=None):
        if train_sample!=None:
            skip = self.sample_skip('./train.csv', train_sample, data_dc['train.csv']['header'])
            for datasetname, dataset_attr in data_dc.iteritems():
                if dataset_attr['train']:
                    dataset_attr['args']['skiprows'] = skip
                #end
            #end
        #end
        train_df = pd.read_csv('./train.csv', **data_dc['train.csv']['args'])
        return train_df
    #end

    def read_test(self, data_dc, test_sample=None):
        if test_sample!=None:
            skip = self.sample_skip('./test.csv', test_sample, data_dc['test.csv']['header'])
            for datasetname, dataset_attr in data_dc.iteritems():
                if not dataset_attr['train']:
                    dataset_attr['args']['skiprows'] = skip
                #end
            #end
        #end
        test_df = pd.read_csv('./test.csv', **data_dc['test.csv']['args'])
        return test_df
    #end

#end

class FeatsManipulator():
    '''
    '''

    def __init__(self):
        pass
    #end

    def low_st_remover(self, df, threshold=0.001):
        '''
        '''
        low_st_sr = df.std()<threshold
        low_st_ls = list(low_st_sr.index[low_st_sr==True])
        df.drop(low_st_ls, axis=1, inplace=True)
        return df
    #end

    def fillna_cat_df(self, df, fill_na='NaN'):
        '''
        Fills missing strings with 'NaN' in categorical type df
        '''
        cat_feat_ls = list(df.dtypes[df.dtypes == "object"].index)
        df[cat_feat_ls].fillna(fill_na, inplace=True)
        return df
    #end

    def fillna_num_df(self, df):
        num_feat_ls = list(df.dtypes[df.dtypes != "object"].index)
        df[num_feat_ls].fillna(df[num_feat_ls].mean(), inplace=True)
        return df
    #end

    def fillna_df(self, df):
        return self.fillna_num_df(self.fillna_cat_df(df))
    #end
    
    def rank_cat_df(self, df, label_sr=None, rank_dc_dc=None):
        '''
        '''
        if ((label_sr is None) and (rank_dc_dc is None)):
            logger.info('No dictionary of target variable supplied!')
            return
        #end
        if rank_dc_dc is None:
            rank_dc_dc = utils.CatRanker().build_df_ranking_dict(df, df[label])
        #end
        df = utils.CatRanker().rank_encode_cat_df(df, label_sr, rank_dc_dc)
        return df, rank_dc_dc
    #end

    def box_cox_df(self, df, skew_threshold=0.2, skewed_feat_ls=None, exclude_ls=None):
        '''
        '''
        if exclude_ls==None:
            exclude_ls = []
        #end
        if skewed_feat_ls==None:
            num_feat_ls = list(df.dtypes[df.dtypes != "object"].index)
            skew_sr = utils.DataFrameDescriber().get_skew(df[num_feat_ls])
            skewed_feat_ls = list(skew_sr[np.absolute(skew_sr)>skew_threshold].index)
        #end
        lam_dc = {}
        skewed_feat_ls = [feat for feat in skewed_feat_ls if feat not in exclude_ls]
        for feat in skewed_feat_ls:
            df[feat] = df[feat] + 1
            df[feat], lam_dc[feat] = boxcox(df[feat])
        #end
        return df, skewed_feat_ls, lam_dc
    #end

    def float_scaler(self, df, exclude_col_ls=None, Scaler_obj=None):
        '''
        '''
        logger.info('Performing scaling of numeric features...')
        if exclude_col_ls==None:
            exclude_col_ls = []
        #end
        col_ls = [col for col in df.columns if col not in exclude_col_ls]
        if Scaler_obj==None:
            Scaler_obj = StandardScaler()
            df[col_ls] = Scaler_obj.fit_transform(df[col_ls])
        else:
            df[col_ls] = Scaler_obj.transform(df[col_ls])
        #end
        return df, Scaler_obj
    #end

    def add_test_flag_and_merge(self, train_df, test_df, flag_type='int'):
        train_df['is_test'] = 0
        test_df['is_test'] = 1
        full_df = pd.concat((train_df, test_df)).reset_index(drop=True)
        full_df['is_test'] = full_df['is_test'].astype(flag_type)
        return full_df
    #end

    def add_leaks(self, test_train_df):
        return test_train_df
    #end

    def preliminary_manipulation(self, df_dc, categorical=True):
        '''
        '''
        train_df = df_dc['train_df']
        test_df = df_dc['test_df']

        logger.info('Performing a preliminary manipulation of the datasets...')
        logger.info('Removing low variance columns..')
        original_feats_ls = list(train_df.columns)
        train_df = self.low_st_remover(train_df, threshold=0.001)
        variance_filtered_feats_ls = list(train_df.columns)
        logger.info('Columns removed:')
        if len(original_feats_ls)>len(variance_filtered_feats_ls):
            logger.info(str([feat for feat in variance_filtered_feats_ls if feat not in original_feats_ls]))
        else:
            logger.info('None')
        #end
        test_df = test_df[[col for col in train_df.columns if col not in ['loss']]]
        logger.info('Filling missing values in train..')
        train_df = self.fillna_df(train_df)
        logger.info('Filling missing values in test..')
        test_df = self.fillna_df(test_df)

        logger.info('Correcting for skewness with box-cox transformation..')
        train_df, skewed_feat_ls, train_lam_dc = self.box_cox_df(train_df, exclude_ls=['loss'])
        test_df, skewed_feat_ls, test_lam_dc = self.box_cox_df(test_df, skewed_feat_ls=skewed_feat_ls, exclude_ls=['loss'])

        logger.info('Manipulating categorical features by ranking..')
        train_df, rank_dc_dc = utils.CatRanker().rank_encode_cat_df(train_df, train_df['loss'])
        logger.info('Ranking complete on train.')
        test_df, rank_dc_dc = utils.CatRanker().rank_encode_cat_df(test_df, train_df['loss'], rank_dc_dc)
        logger.info('Ranking complete on test.')

        logger.info('Scaling features..')
        train_df, Scaler_obj = self.float_scaler(train_df, exclude_col_ls=['id', 'loss'], Scaler_obj=None)
        test_df, _ = self.float_scaler(test_df, exclude_col_ls=['id', 'loss'], Scaler_obj=Scaler_obj)

        train_test_df = self.add_test_flag_and_merge(train_df, test_df, flag_type='int')
        train_test_df = self.add_leaks(train_test_df)

        logger.info('Preliminary manipulation of datasets performed.')
        train_test_df = train_test_df.sort_values(by='id')
        return train_test_df
    #end

#end

class PrelFeatsSelector():
    '''
    '''
    def __init__(self, SelectorModel, num_threshold=100, cat_threshold=10, sample_dc=None, best=False):
        self.SelectorModel = SelectorModel
        self.n_thresh = num_threshold
        self.c_thresh = cat_threshold
        self.best = best
        self.sample_dc = self.build_sample_dc(sample_dc)
    #end

    def build_sample_dc(self, sample_dc):
        if sample_dc==None:
            sample_dc = {}
            sample_dc['numeric'], sample_dc['categorical'] = 1.0, 1.0
            logger.info('Preliminary feature selection will be performed without downsampling of datasets.')
        else:
            logger.info('Sampling for feature selection: %s numeric and %s categorical' % (str(sample_dc['numeric']), str(sample_dc['categorical'])))
        #end
        return sample_dc
    #end

    def num_feats_selector(self, feats_df, label_id, threshold):
        '''
        '''
        #score = make_scorer(MCC, greater_is_better=True)
        rfecv = RFECV(estimator=self.SelectorModel, step=1, cv=StratifiedKFold(3), scoring='neg_mean_absolute_error')
        feat_ls = [col for col in feats_df.columns if col!=label_id]
        feats_df = feats_df.sample(frac=self.sample_dc['numeric'])
        feat_ar = np.array(feats_df[feat_ls])
        label_ar = np.log(feats_df[label_id].ravel() + 200.0)
        rfecv.fit(feat_ar, label_ar)
        if self.best:
            ranked_feats_idx = [idx for idx, rank in enumerate(rfecv.ranking_) if rank==1]
            print rfecv.grid_scores_
            sys.stdout.flush()
        else:
            ranked_feats_idx = sorted(range(len(rfecv.ranking_)), key=lambda k: rfecv.ranking_[k])[0:threshold]
        #end
        ranked_feats_ls = [feat_ls[idx] for idx in ranked_feats_idx]
        return ranked_feats_ls 
    #end

    def cat_feats_selector(self, feats_df, label_id, threshold, return_scores=False):
        '''
        '''
        feat_ls = [col for col in feats_df.columns if col!=label_id]
        score = make_scorer(MCC, greater_is_better=True)
        feat_score_ls = []
        feats_df = feats_df.sample(frac=self.sample_dc['categorical'])
        for fn, feat in enumerate(feat_ls):
            if (10*fn)%int(10*np.round((len(feat_ls)/10.0)))==0:
                logger.info('Progress: %s %%' % (str((100*fn)/int(10*np.round((len(feat_ls)/10.0))-1))))
            #end
            fitted_le_dc, fitted_ohe = CatEncoder().fit_onehot_to_cat(feats_df[[feat]])
            encoded_sparse_arr = CatEncoder().transform_onehot(feats_df[[feat]], fitted_le_dc, fitted_ohe)
            cv_scores = cross_val_score(self.SelectorModel, encoded_sparse_arr, feats_df[label_id].ravel(), cv=StratifiedKFold(3), scoring=score)
            feat_score_ls.append((feat, cv_scores.mean()))
        #end
        rank_ls = [el[0] for el in sorted(feat_score_ls, key=lambda tup: tup[1], reverse=True)]
        score_ls = [el[1] for el in sorted(feat_score_ls, key=lambda tup: tup[1], reverse=True)]
        if return_scores:
            return rank_ls, score_ls
        else:
            return rank_ls[0:threshold]
        #end
    #end

    def select_feats(self, feats_df, label_id='response', feat_type='numeric'):
        logger.info('Performing features selection...')
        logger.info('Number of features to be selected: %s numeric, %s categorical' % (str(self.n_thresh), str(self.c_thresh)))
        logger.info('Downsampling for speed: %s numeric, %s cateforical' % (str(self.sample_dc['numeric']), str(self.sample_dc['categorical'])))
        if feat_type=='numeric':
            logger.info('Fitting Recursive Feature Elimination Model for numeric feature selection...')
            ranked_feats_ls = self.num_feats_selector(feats_df, label_id, self.n_thresh)
        elif feat_type=='categorical':
            logger.info('Performing feature ranking for categorical features...')
            ranked_feats_ls = self.cat_feats_selector(feats_df, label_id, self.c_thresh)
        #end
        logger.info('Feature ranking complete!')
        return ranked_feats_ls
    #end

#end


class Assembler():
    '''
    '''


#end

class Regressor():
    '''
    '''
    def __init__(self):
        pass
    #end

    def label_transform(self, label_sr, inverse=False):
        '''
        '''
        shift = 200.0
        if inverse:
            transf_label = np.exp(label_sr) - shift
        else:
            transf_label = np.log(label_sr + shift)
        #end
        return transf_label
    #end

    def kfolds_cv(self, train_df, label_sr, test_df, parameter_dc, n_folds):
        '''
        '''
        logger.info('Performing CV...')
        d_test = xgb.DMatrix(test_df)
        cv_dc = dict()
        xgb_rounds = list()
        kf = KFold(train_df.shape[0], n_folds=n_folds)
        transf_label_sr = self.label_transform(label_sr)

        for ifold, (train_index, test_index) in enumerate(kf):
            logger.info('Training Fold: ' + str(ifold + 1))
            X_train, X_val = train_df.iloc[train_index], train_df.iloc[test_index]
            y_train, y_val = transf_label_sr[train_index], transf_label_sr[test_index]

            d_train = xgb.DMatrix(X_train, label=y_train.ravel())
            d_valid = xgb.DMatrix(X_val, label=y_val.ravel())
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]

            booster = xgb.train(parameter_dc['params'], d_train, 
                                parameter_dc['cross_val']['num_boost_round'], 
                                watchlist,
                                early_stopping_rounds=parameter_dc['cross_val']['early_stopping_rounds'])

            xgb_rounds.append(booster.best_iteration)
            scores_val = booster.predict(d_valid, ntree_limit=booster.best_ntree_limit)
            cv_score = mean_absolute_error(self.label_transform(y_val, inverse=True), self.label_transform(scores_val, inverse=True))
            logger.info('Evaluation MAE: %.6f' % cv_score)
            y_pred = self.label_transform(booster.predict(d_test, ntree_limit=booster.best_ntree_limit), inverse=True)
            if ifold==0:
                fpred = y_pred
                cv_sum = 0.0
            else:
                fpred = pred + y_pred
            #end
            pred = fpred
            cv_sum = cv_sum + cv_score
        #end

        mpred = 1.0 * pred / n_folds
        score = 1.0 * cv_sum / n_folds
        logger.info('Average evalluation MAE: %.6f' % score)
        cv_dc['avg_n_rounds'] = int(np.mean(xgb_rounds))
        cv_dc['avg_pred_ls'] = mpred

        return cv_dc
    #end

    def regress(self, train_df, label_sr, test_df, parameter_dc):
        logger.info('Training data shape: ' + str(train_df.shape))
        logger.info('Test data shape: ' + str(test_df.shape))
        logger.info('Learning...')

        transf_label_sr = self.label_transform(label_sr)
        dtrain = xgb.DMatrix(train_df, label=transf_label_sr.ravel())
        dtest = xgb.DMatrix(test_df)

        logger.info('Training...')
        booster = xgb.train(parameter_dc['params'], dtrain, parameter_dc['addtn_par']['num_boost_round'])
        logger.info('Saving model...')
        pickle.dump(booster, open("trained_booster.dat", "wb"))
        logger.info('Predicting...')
            
        prediction_ls =self.label_transform(booster.predict(dtest), inverse=True)
  
        return prediction_ls
    #end

#end

class Ensembler():
    '''
    '''
    def __init__(self):
        pass
    #end

    def get_beta_pdf_from_performance_list(self, performance_ls, alpha=0.5, beta=0.5, reverse_is_better=False):
        '''
        '''
        if reverse_is_better:
            perf_ls = [-1.0*el for el in performance_ls]
        else:
            perf_ls = list(performance_ls)
        #end
        worse_perf = np.min(perf_ls)
        best_perf = np.max(perf_ls)
        gap = best_perf - worse_perf
        relative_perf_ls = [1.0*(performance - worse_perf)/(gap) for performance in perf_ls]
        beta_pdf_ls = [scipy.stats.beta.cdf(rel_perf, alpha, beta) for rel_perf in relative_perf_ls]
        return beta_pdf_ls
    #end

    def scale_weight_ls(self, weight_ls, non_zero_weight=True):
        '''
        '''
        adj_weight_ls = list(weight_ls)
        if non_zero_weight:
            smallest_nonzero = np.min([w for w in weight_ls if w>0.0])
            adj_weight_ls = [el if el>0.0 else smallest_nonzero for el in adj_weight_ls]
        #end
        scaled_ls = [1.0*aw/np.sum(adj_weight_ls) for aw in adj_weight_ls]
        return scaled_ls
    #end

    def get_ensembled_output(self, result_ls_ls, weight_ls=None, performance_ls=None):
        '''
        '''
        n_models = len(result_ls_ls)
        if not weight_ls and not performance_ls:
            weight_ls = [1.0/n_models for model in range(n_models)]
        elif not weight_ls:
            scaled_weight_ls = self.scale_weight_ls(get_beta_pdf_from_performance_list(performance_ls))
        #end
        scaled_weight_ls = self.scale_weight_ls(weight_ls)
        ens_output_ls = np.dot(np.transpose(result_ls_ls), scaled_weight_ls)
        return ens_output_ls
    #end

#end



class ModelOptimizer():
    '''
    '''
    def __init__(self):
        pass
    #end

#end

class Stacker():
    '''
    '''
    def __init__(self, par_dc):
        self.par_dc = par_dc
        return
    #end

    def stack(train_df, label_sr, n_folds=10):
        cv_dc = Regressor().kfolds_cv(train_df, label_sr, train_df, self.par_dc, n_folds)
        return cv_dc['avg_pred_ls']
    #end

#end

class OutputHandler():
    '''
    '''

    def __init__(self):
        pass
    #end

    def output_writer(self, id_col, predictions, header_dc=None, data_path='./', name=None, gz=True):
        '''
        '''
        if not name:
            name = 'subm'
        #end
        if not header_dc:
            header_dc = {'id': 'id', 'label': 'label'}
        #end
        timestamp_str = str(datetime.now().strftime('%Y-%m-%d-%H-%M'))
        output_df = pd.DataFrame({header_dc['id'] : id_col, header_dc['label']: predictions})
        if gz:
            output_df.to_csv(data_path  + timestamp_str + name + '.gz', index = False, compression='gzip')
            saveformat = '.gz'
        else:
            output_df.to_csv(data_path + timestamp_str + name + '.csv', index = False)
            saveformat = '.csv'
        #end
        logger.info('Data successfully saved as %s' % saveformat)
        return
    #end

#end

