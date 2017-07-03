"""
  **************************************
  Created by Romano Foti - rfoti
  On 09/15/2016
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
import requests
import base64
import zipfile

#******************************************************************************
# Defining functions
#******************************************************************************

#******************************************************************************

#******************************************************************************
# Defining classes
#******************************************************************************

class KaggleRequest():
    '''
    Connects to Kaggle and downloads datasets.
    '''

    def __init__(self, credentials_file=None, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger
        if not credentials_file:
            self.credentials_file = './kaggle_cred.cred'
        #end
    #end

    def decrypt(self, credentials_file):
        '''
        This function retrieves the encrypted credential file
        and returns a dictionary with decripted username and password
        '''
        cred_file = open(credentials_file, 'r')
        cred_lines_encry_ls = cred_file.read().split(',')
        try:
            creds_dc = {'UserName': base64.b64decode(cred_lines_encry_ls[0]), 
                        'Password': base64.b64decode(cred_lines_encry_ls[1])}
        except:
            if self.verbose:
                if not self.logger:
                    print 'Problem decrypting credentials. Request terminated.'
                    sys.stdout.flush()
                else:
                    self.logger.info('Problem decrypting credentials. Request terminated.')
                #end
            #end
            return
        #end
        return creds_dc
    #end

    def unzip(self, filename):
        '''
        Unzips a file
        '''
        output_path = '/'.join([level for level in filename.split('/')[0:-1]]) + '/'
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(output_path)
        #end
        z.close()
        if self.verbose:
            if not self.logger:
                print 'File successfully unzipped!'
                sys.stdout.flush()
            else:
                self.logger.info('File successfully unzipped!')
            #end
        #end
        return
    #end

    def retrieve_dataset(self, data_url, local_filename=None, chunksize=512, unzip=True):
        '''
        Connects to Kaggle website, downloads the dataset one chunk at a time
        and saves it locally.
        '''
        if not data_url:
            if self.verbose:
                if not self.logger:
                    print 'A data URL needs to be provided.'
                    sys.stdout.flush()
                else:
                    self.logger.info('A data URL needs to be provided.')
                #end
            #end
        if not local_filename:
            try:
                local_filename = './' + data_url.split('/')[-1]
                if self.verbose:
                    if not self.logger:
                        print 'Dataset name inferred from data_url. It is going to be saved in the default location.'
                        sys.stdout.flush()
                    else:
                        self.logger.info('Dataset name inferred from data_url. It is going to be saved in the default location.')
                    #end
                #end
            except:
                if self.verbose:
                    if not self.logger:
                        print 'Could not infer data name, request terminated.'
                        sys.stdout.flush()
                    else:
                        self.logger.info('Could not infer data name, request terminated.')
                    #end
                #end
                return
            #end
        #end
        kaggle_info = self.decrypt(self.credentials_file)
        chunks = chunksize * 1024
        req = requests.get(data_url) # attempts to download the CSV file and gets rejected because we are not logged in
        req = requests.post(req.url, data=kaggle_info, stream=True) # login to Kaggle and retrieve the data
        f = open(local_filename, 'w')
        for chunk in req.iter_content(chunk_size=chunks): # Reads one chunk at a time into memory
            if chunk: # Filtering out keep-alive new chunks
                f.write(chunk)
            #end
        #end
        f.close()
        if self.verbose:
            if not self.logger:
                print 'Data successfully downloaded!'
                sys.stdout.flush()
            else:
                self.logger.info('Data successfully downloaded!')
            #end
        #end
        if unzip:
            self.unzip(local_filename)
        #end
        return
    #end

#end
