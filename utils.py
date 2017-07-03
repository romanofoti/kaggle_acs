"""
  **************************************
  Created by Romano Foti - rfoti
  On 10/22/2016
  *************************************
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
import random
import logging
from datetime import datetime
from scipy.stats import skew, boxcox
import matplotlib.pyplot as plt
import seaborn as sns #seaborn wrapper plotting library
#-----------------------------
# User defined modules and functions
#-----------------------------



#-----------------------------
# Logging
#-----------------------------

class Logging():
    '''
    '''
    def __init__(self):
        pass
    #end

    def configure_logger(self, logname, logfile):
        """ Configures a logger object, and adds a stream handler as well as a 
        file handler. """
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    #end

#end

class Timing():
    '''
    '''
    def __init__(self):
        pass
    #end

    def timer(self, start_time=None, logger=None):
        '''
        '''
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
            if not logger:
                print(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
                sys.stdout.flush()
            else:
                logger.info(' Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
            #end
        #end
    #end

#end

class DataFrameDescriber():
    '''
    '''
    def __init__(self, df=None):
        if not isinstance(df, pd.DataFrame):
            pass
        else:
            self.perc_miss = self.get_percent_missing(df);
            self.miss_loc = self.get_miss_loc(df)
            self.skewed_feats_sr = self.get_skew(df)
        #end
        return
    #end

    def log_or_print(self, message, logger=None):
        if not logger:
            print(message)
            sys.stdout.flush()
        else:
            logger.info(message)
        #end
        return
    #end

    def get_skew(self, df):
        numeric_feat_ls = list(df.dtypes[df.dtypes != "object"].index)
        skewed_feats_sr = df[numeric_feat_ls].apply(lambda x: skew(x.dropna()))
        return skewed_feats_srs
    #end

    def get_miss_loc(self, df):
        return pd.isnull(df).any(1).nonzero()[0] #create an array with the index of the rows where missing data are present
    #end

    def get_percent_missing(self, df):
        inds = self.get_miss_loc(df) #retrieve the index of rows containing missing values
        return 1.0*len(inds)/len(df) #return percent of rows containing missing values
    #end

    def describe(self, df, logger=None):
        self.perc_miss = self.get_percent_missing(df);
        self.miss_loc = self.get_miss_loc(df)
        self.skewed_feats_sr = self.get_skew(df)
        self.log_or_print('Numeric column description:')
        self.log_or_print(df.describe(), logger) #describe the dataset
        self.log_or_print('Percent of rows containing missing data: %.1f' %self.perc_miss, logger)
        self.log_or_print('Description of the DataFrame by filtering out rows that DO NOT contain missing data:', logger)
        self.log_or_print(df.iloc[self.get_miss_loc(df)].describe())
        self.log_or_print('Skewness of numerci features:')
        self.log_or_print(self.get_skew(df), logger)
        self.log_or_print('To retrieve percent missing: DataFrameDescriber(df).perc_miss', logger)
        self.log_or_print('To retrieve index of rows with missing values: DataFrameDescriber(df).miss_loc', logger)
        self.log_or_print('To retrieve skewness of numeric feats: DataFrameDescriber(df).get_skew', logger)

        return
    #end

#end

class EDAPlotter():
    '''
    '''
    def __init__(self):
        pass
    #end

    def heatmap_corr(self, df, sample=10000, savefig=False):
        '''
        Creates a heatmap of correlation from a dataframe
        '''    
        plt.figure() #initialize the figure
        corrmat = df.sample(sample).corr() # build the matrix of correlation from the dataframe after sampling
        f, ax = plt.subplots(figsize=(12, 9))   # set up the matplotlib figure
        sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True); # draw the heatmap using seaborn
        if savefig:
            fig.savefig('./heatmap_corr.png')
        #end
        return
    #end

    def scatter_corr(self, df, response_label, sample=10000, savefig=False):
        num_df = df[list(df.dtypes[df.dtypes != "object"].index)].sample(sample) #extract numerical features and sample down
        n_feats = num_df.shape[1]
        display = 10
        n_plots = (n_feats / display) + 1
        for iplot in xrange(n_plots):
            fig = plt.figure(figsize=(16,16))
            locs = np.hstack(([0], range(iplot*display+1, (iplot+1)*display)))
            sns.pairplot(num_df.iloc[:,locs], diag_kind='kde', hue=response_label, palette='Set1');
            if savefig:
                fig.savefig('./scatter_corr_' + str(iplot) + '.png')
            #end
        #end
        return
    #end

    def label_regression_stack(self, df, response_label, sample=10000, savefig=False):
        '''
        '''
        plt.figure()
        num_feat_ls = list(df.dtypes[df.dtypes != "object"].index)
        num_df = df[num_feat_ls].sample(sample)
        n_feats = num_df.shape[1]
        display = 5
        n_plots = (n_feats / display) + 1
        for iplot in xrange(n_plots):
            fig = plt.figure()
            subplot_ls = range(iplot*display, min(n_feats, (iplot+1)*display))
            sns.pairplot(data=num_df, x_vars=[num_feat_ls[el] for el in subplot_ls], y_vars=response_label);
            if savefig:
                fig.savefig('./scatter_reg_' + str(iplot) + '.png')
            #end
        #end
        return
    #end

#end

class CatRanker():

    def __init__(self):
        pass
    #end

    def build_sr_ranking_dict(self, cat_feat_sr, label_sr):
        '''
        Returns a dictionary with the rank of each feature
        encountered in the cat_feat_sr
        '''
        unique_ls = list(cat_feat_sr.unique())
        rank_dc = {}
        for cat in unique_ls:
            rank_dc[cat] = label_sr[cat_feat_sr==cat].mean()
        #end
        return rank_dc
    #end

    def assign_sr_rank(self, cat_feat_sr, label_sr, rank_dc):
        '''
        Assigns the rank to each categorical value found in the 
        given pandas Series (cat_feat_sr). It needs the rank dictionary
        as input
        '''
        unique_ls = list(cat_feat_sr.unique())
        for cat in unique_ls:
            if cat in rank_dc.keys():
                cat_feat_sr[cat_feat_sr==cat] = rank_dc[cat]
            else:
                cat_feat_sr[cat_feat_sr==cat] = label_sr.mean()
            #end
        #end
        return cat_feat_sr
    #end

    def build_df_ranking_dict(self, df, label_sr):
        '''
        Returns a dictionary of dictionaries
        Firs tier dictionary: keys are the categorical columns
        Second tier dictionary: keys are the categories encountered
                                while values are the corresponding
                                ranks
        '''
        cat_df = df[list(df.dtypes[df.dtypes == "object"].index)]
        rank_dc_dc = {}
        c_feat_ls = list(cat_df.columns)
        for c_feat in c_feat_ls:
            rank_dc_dc[c_feat] = self.build_sr_ranking_dict(cat_df[c_feat], label_sr)
        #end
        return rank_dc_dc
    #end

    def rank_encode_cat_df(self, df, label_sr, rank_dc_dc=None):
        '''
        Assigns the rank to each categorical value found in each
        of the categorical colum on a pandas DataFrame (df).
        It needs the ranking dictionary of dictionaries as inputt
        '''
        cat_df = df[list(df.dtypes[df.dtypes == "object"].index)]
        c_feat_ls = list(cat_df.columns)
        if rank_dc_dc is None:
            rank_dc_dc = self.build_df_ranking_dict(df, label_sr)
        #end
        for c_feat in c_feat_ls:
            df[c_feat] = self.assign_sr_rank(df[c_feat], label_sr, rank_dc_dc[c_feat])
        #end
        return df, rank_dc_dc
    #end

#end

class CatEncoder():
    '''
    '''

    def __init__(self):
        pass
    #end

    def encode_labels(self, df, cols2encode_ls=None, encode_1by1=False):
        if not cols2encode_ls:
            cols2encode_ls = list(df.select_dtypes(include=['category','object']))
        #end
        if encode_1by1:
            le = LabelEncoder()
            for feature in cols2encode_ls:
                try:
                    transformed_df[feature] = le.fit_transform(df[feature])
                except:
                    print('Error encoding '+ feature)
                #end
            #end
            return le, transformed_df
        else:
            le_dc = defaultdict(LabelEncoder)
            transformed_df = df.apply(lambda x: le_dc[x.name].fit_transform(x))
            return le_dc, transformed_df
        #end
    #end

    def fit_onehot_to_cat(self, df, cols2encode_ls=None):
        if not cols2encode_ls:
            cols2encode_ls = list(df.select_dtypes(include=['category','object']))
        #end
        le_dc, le_df = self.encode_labels(df, cols2encode_ls=cols2encode_ls, encode_1by1=False) #call label encoder
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe = ohe.fit(le_df, cols2encode_ls)
        return le_dc, ohe
    #end

    def transform_onehot(self, df, le_dc, ohe):
        le_df = df.apply(lambda x: le_dc[x.name].transform(x))
        return ohe.transform(le_df)
    #end

#end

#-----------------------------
# SWAP ROWS and COLUMNS
#-----------------------------

def swap_rows(nparray, frm, to, return_array=False):
    nparray[[frm, to],:] = nparray[[to, frm],:] #swaps rows using advanced slicing
    if return_array:
      return nparray
    else:
      return
    #end
#end
    
def swap_cols(nparray, frm, to, return_array=False):
    nparray[:,[frm, to]] = nparray[:,[to, frm]] #swaps columns using advanced slicing
    if return_array:
      return nparray
    else:
      return
    #end
#end

#-----------------------------