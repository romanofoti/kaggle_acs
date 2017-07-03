
"""
  **************************************
  Created by Romano Foti - rfoti
  On 12/08/2016
  **************************************
"""

#******************************************************************************
# Importing packages
#******************************************************************************
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

import utils
import kaggle_utils

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

class OutputHandler():
    '''
    '''

    def __init__(self):
        pass
    #end

    def output_writer(self, id_col, predictions, data_path='./', gz=True):
        output_df = pd.DataFrame({ 'id' : id_col, 'loss': predictions})
        if gz:
            output_df.to_csv(data_path + '/submission.gz', index = False, compression='gzip')
            saveformat = '.gz'
        else:
            output_df.to_csv(data_path + '/submission.csv', index = False)
            saveformat = '.csv'
        #end
        logger.info('Data successfully saved as %s' % saveformat)
        return
    #end

#end

shift = 200
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)
def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain



#******************************************************************************
# MAIN PROGRAM
#******************************************************************************
print('\nStarted')
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

logger = utils.Logging().configure_logger('acs_kivara_logs', './acs_kivara_logfile.log')

DataDownloader().download_from_kaggle(url_dc)

train_sample = None #None if no sampling is required
test_sample = None #None if no sampling is required

#******************************************************************************

train = DataReader().read_train(data_dc, train_sample=train_sample)
test = DataReader().read_test(data_dc, test_sample=test_sample)

numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
train_test, ntrain = mungeskewed(train, test, numeric_feats)

logger.info('Starting categorical column manipulation...')
# taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
for column in list(train.select_dtypes(include=['object']).columns):
    if train[column].nunique() != test[column].nunique():
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)


        def filter_cat(x):
            if x in remove:
                return np.nan
            return x


        train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

# taken from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)
train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25
logger.info('Categorical column manipulation ended.')

logger.info('Starting features combinations...')
print('')
for comb in itertools.combinations(COMB_FEATURE, 2):
    feat = comb[0] + "_" + comb[1]
    train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
    train_test[feat] = train_test[feat].apply(encode)
    print('Combining Columns:', feat)

print('')
for col in categorical_feats:
    print('Analyzing Column:', col)
    train_test[col] = train_test[col].apply(encode)

print(train_test[categorical_feats])
logger.info('Features combinations ended.')

logger.info('Scaling...')
ss = StandardScaler()
train_test[numeric_feats] = \
    ss.fit_transform(train_test[numeric_feats].values)
logger.info('Scaled.')

train = train_test.iloc[:ntrain, :].copy()
test = train_test.iloc[ntrain:, :].copy()

print('\nMedian Loss:', train.loss.median())
print('Mean Loss:', train.loss.mean())

ids = pd.read_csv('./test.csv')['id']
train_y = np.log(train['loss'] + shift)
train_x = train.drop(['loss','id'], axis=1)
test_x = test.drop(['loss','id'], axis=1)

n_folds = 8
cv_sum = 0
early_stopping = 100
fpred = []
xgb_rounds = []

d_train_full = xgb.DMatrix(train_x, label=train_y)
d_test = xgb.DMatrix(test_x)

logger.info('Starting CV...')
kf = KFold(train.shape[0], n_folds=n_folds)
for i, (train_index, test_index) in enumerate(kf):
    logger.info('\n Fold %d' % (i+1))
    X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
    y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]

    rand_state = 2016

    params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.04,
        'objective': 'reg:linear',
        'max_depth': 12,
        'eta': 0.04,
        'min_child_weight': 100,
        'booster': 'gbtree'}

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]

    clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=20,
                    obj=fair_obj,
                    feval=xg_eval_mae)

    xgb_rounds.append(clf.best_iteration)
    scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
    logger.info('eval-MAE: %.6f' % cv_score)
    y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred
    cv_sum = cv_sum + cv_score

mpred = pred / n_folds
score = cv_sum / n_folds
logger.info('Average eval-MAE: %.6f' % score)
n_rounds = int(np.mean(xgb_rounds))

logger.info("Writing results")
result = pd.DataFrame(mpred, columns=['loss'])
result["id"] = ids
result = result.set_index("id")
logger.info("%d-fold average prediction:" % n_folds)

now = datetime.now()
score = str(round((cv_sum / n_folds), 6))
sub_file = 'submission_5fold-average-xgb_fairobj_' + str(score) + '_' + str(
    now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
logger.info("Writing submission: %s" % sub_file)
result.to_csv(sub_file, index=True, index_label='id')

OutputHandler().output_writer(ids.tolist(), mpred, gz=True)