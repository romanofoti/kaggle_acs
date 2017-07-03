"""
  **************************************
  Created by Romano Foti - rfoti
  On 12/02/2016
  **************************************
"""

#******************************************************************************
# Importing packages
#******************************************************************************
import numpy as np
import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

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

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

#******************************************************************************
# MAIN PROGRAM
#******************************************************************************

np.random.seed(123)
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

index = list(train.index)
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = np.log(train['loss'].values+200)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

#******************************************************************************

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

#******************************************************************************

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

#******************************************************************************

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)
#end

#******************************************************************************

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 10
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  validation_data=(xte.todense(), yte), 
                                  verbose = 0)
        temp = np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200
        pred += temp
        logger.info("Fold val bagging score after" + str(j+1) + "rounds is: " + str(mean_absolute_error(np.exp(yte)-200, pred/(j+1))))
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-200
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte)-200, pred)
    i += 1
    logger.info('Fold ' + str(i) + '- MAE: ' + str(score))
#end

logger.info('Total - MAE:' + str(mean_absolute_error(np.exp(y)-200, pred_oob)))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('./preds_oob.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('./submission_keras_shift_perm.csv', index = False)

OutputHandler().output_writer(id_test.tolist(), pred_test, name='_pred_test', gz=True)
OutputHandler().output_writer(id_test.tolist(), pred_oob, name='pred_train', gz=True)

#******************************************************************************