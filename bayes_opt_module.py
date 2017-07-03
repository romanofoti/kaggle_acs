
"""
  **************************************
  Created by Romano Foti - rfoti
  On 12/09/2016
  **************************************
"""

#******************************************************************************
# Importing packages
#******************************************************************************




    start_time = timer(None)
    print("\n# Global Optimization Search for SVR Parameters C and gamma\n")
    print("\n Please note that negative RMSE values will be shown below. This is because")
    print(" RMSE needs to be minimized, while Bayes optimizer always maximizes the function.\n")

    svrBO = BayesianOptimization(svrcv, {
                                         'log2C': (-6, 6),
                                         'log2gamma': (-12, 0)
                                        })

    svrBO.maximize(init_points=50, n_iter=150, acq="ei", xi=0.0)
    print("-" * 53)
    timer(start_time)

    best_RMSE = round((-1.0 * svrBO.res['max']['max_val']), 6)
    C = svrBO.res['max']['max_params']['log2C']
    gamma = svrBO.res['max']['max_params']['log2gamma']

    print("\n Best RMSE value: %f" % best_RMSE)
    print(" Best SVR parameters:  log2(C) = %f  log2(gamma) = %f" % (C, gamma))

    start_time = timer(None)
    print("\n# Making Prediction")

    svr = SVR(kernel='rbf', C=math.pow(2,C), gamma=math.pow(2,gamma), cache_size=2000, verbose=False, max_iter=-1, shrinking=False)

    x_true = np.array(train['SalePrice'])
    x_pred = np.exp(cross_val_predict(svr, X=train_data, y=target, cv=folds))




    def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(" Time taken: %i minutes and %s seconds." % (tmin, round(tsec,2)))

def svrcv(log2C, log2gamma):
    cv_score = cross_val_score(
        SVR
        (
        C=math.pow(2,log2C),
        kernel='rbf',
        gamma=math.pow(2,log2gamma),
        cache_size=2000,
        verbose=False,
        max_iter=-1,
        shrinking=False,
        ),
        train_data,
        target,
        'mean_squared_error',
        cv=folds
        ).mean()
    return ( -1.0 * np.sqrt(-cv_score))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def sparse_df_to_array(df):
    num_rows = df.shape[0]
    data = []
    row = []
    col = []
    for i, col_name in enumerate(df.columns):
        if isinstance(df[col_name], pd.SparseSeries):
            column_index = df[col_name].sp_index
            if isinstance(column_index, BlockIndex):
                column_index = column_index.to_int_index()
            ix = column_index.indices
            data.append(df[col_name].sp_values)
            row.append(ix)
            col.append(len(df[col_name].sp_values) * [i])
        else:
            data.append(df[col_name].values)
            row.append(np.array(range(0, num_rows)))
            col.append(np.array(num_rows * [i]))
    data_f = np.concatenate(data)
    row_f = np.concatenate(row)
    col_f = np.concatenate(col)
    arr = coo_matrix((data_f, (row_f, col_f)), df.shape, dtype=np.float64)
    return arr.tocsr()

if __name__ == "__main__":

    folds = 5

    CV = CountVectorizer(min_df=1, analyzer='word', max_features=30000, binary=True)

    print("\n Please read the comments carefully. The code can be much shorter if you follow directions.")

# Load data set and target values

    start_time = timer(None)
    print("\n# Reading and Processing Data")
    train = pd.read_csv("../input/train.csv", dtype = {'Id' : np.str} )
    print("\n Initial Train Set Matrix Dimensions: %d x %d" % (train.shape[0], train.shape[1]))
    target = np.log(np.array(train['SalePrice']))
    test = pd.read_csv("../input/test.csv", dtype = {'Id' : np.str} )
    print("\n Initial Test Set Matrix Dimensions: %d x %d" % (test.shape[0], test.shape[1]))
    ids = test['Id']

# Minimal feature engineering - this can be done much better
    train['Age'] = train['YrSold'] - train['YearBuilt']
    test['Age'] = train['YrSold'] - train['YearBuilt']
    train['AgeRemod'] = train['YrSold'] - train['YearRemodAdd']
    test['AgeRemod'] = train['YrSold'] - train['YearRemodAdd']
    train['Baths'] = train['FullBath'] + train['HalfBath']
    test['Baths'] = test['FullBath'] + test['HalfBath']
    train['BsmtBaths'] = train['BsmtFullBath'] + train['BsmtHalfBath']
    test['BsmtBaths'] = test['BsmtFullBath'] + test['BsmtHalfBath']

# Feature fixing - converting numerical to categorical, removing spaces from features, coverting single-letter features
    train['MSSubClass'] = train['MSSubClass'].astype(str)
    test['MSSubClass'] = test['MSSubClass'].astype(str)
    train['Alley'].replace('NA', 'NoAlley', inplace=True)
    test['Alley'].replace('NA', 'NoAlley', inplace=True)
    train['MSZoning'].replace('A', 'AG', inplace=True)
    train['MSZoning'].replace('C', 'CO', inplace=True)
    train['MSZoning'].replace('I', 'IN', inplace=True)
    test['MSZoning'].replace('A', 'AG', inplace=True)
    test['MSZoning'].replace('C', 'CO', inplace=True)
    test['MSZoning'].replace('I', 'IN', inplace=True)
    train['HouseStyle'].replace('1.5Fin', '1_5Fin', inplace=True)
    train['HouseStyle'].replace('1.5Unf', '1_5Unf', inplace=True)
    train['HouseStyle'].replace('2.5Fin', '2_5Fin', inplace=True)
    train['HouseStyle'].replace('2.5Unf', '2_5Unf', inplace=True)
    test['HouseStyle'].replace('1.5Fin', '1_5Fin', inplace=True)
    test['HouseStyle'].replace('1.5Unf', '1_5Unf', inplace=True)
    test['HouseStyle'].replace('2.5Fin', '2_5Fin', inplace=True)
    test['HouseStyle'].replace('2.5Unf', '2_5Unf', inplace=True)
    train['RoofMatl'].replace('Tar&Grv', 'Tar_Grv', inplace=True)
    test['RoofMatl'].replace('Tar&Grv', 'Tar_Grv', inplace=True)
    train['Exterior1st'].replace('Wd Sdng', 'WdSdng', inplace=True)
    test['Exterior1st'].replace('Wd Sdng', 'WdSdng', inplace=True)
    train['Exterior2nd'].replace('Wd Sdng', 'WdSdng', inplace=True)
    test['Exterior2nd'].replace('Wd Sdng', 'WdSdng', inplace=True)
    train['MasVnrType'].replace('None', 'NoVen', inplace=True)
    test['MasVnrType'].replace('None', 'NoVen', inplace=True)
    train['BsmtQual'].replace('NA', 'NB', inplace=True)
    test['BsmtQual'].replace('NA', 'NB', inplace=True)
    train['BsmtCond'].replace('NA', 'NB', inplace=True)
    test['BsmtCond'].replace('NA', 'NB', inplace=True)
    train['BsmtExposure'].replace('NA', 'NB', inplace=True)
    test['BsmtExposure'].replace('NA', 'NB', inplace=True)
    train['BsmtFinType1'].replace('NA', 'NB', inplace=True)
    test['BsmtFinType1'].replace('NA', 'NB', inplace=True)
    train['BsmtFinType2'].replace('NA', 'NB', inplace=True)
    test['BsmtFinType2'].replace('NA', 'NB', inplace=True)
    train['FireplaceQu'].replace('NA', 'NF', inplace=True)
    test['FireplaceQu'].replace('NA', 'NF', inplace=True)
    train['GarageType'].replace('NA', 'NG', inplace=True)
    test['GarageType'].replace('NA', 'NG', inplace=True)
    train['GarageFinish'].replace('NA', 'NG', inplace=True)
    test['GarageFinish'].replace('NA', 'NG', inplace=True)
    train['GarageQual'].replace('NA', 'NG', inplace=True)
    test['GarageQual'].replace('NA', 'NG', inplace=True)
    train['GarageCond'].replace('NA', 'NG', inplace=True)
    test['GarageCond'].replace('NA', 'NG', inplace=True)
    train['PoolQC'].replace('NA', 'NP', inplace=True)
    test['PoolQC'].replace('NA', 'NP', inplace=True)
    train['Fence'].replace('NA', 'NF', inplace=True)
    test['Fence'].replace('NA', 'NF', inplace=True)
    train['MiscFeature'].replace('NA', 'NoF', inplace=True)
    test['MiscFeature'].replace('NA', 'NoF', inplace=True)
    train['CentralAir'].replace('Y', 'Yes', inplace=True)
    train['CentralAir'].replace('N', 'No', inplace=True)
    test['CentralAir'].replace('Y', 'Yes', inplace=True)
    test['CentralAir'].replace('N', 'No', inplace=True)
    train['PavedDrive'].replace('Y', 'Yes', inplace=True)
    train['PavedDrive'].replace('N', 'No', inplace=True)
    train['PavedDrive'].replace('P', 'Partial', inplace=True)
    test['PavedDrive'].replace('Y', 'Yes', inplace=True)
    test['PavedDrive'].replace('N', 'No', inplace=True)
    test['PavedDrive'].replace('P', 'Partial', inplace=True)

# Feature consolidation
    train['Conditions'] = train[['Condition1', 'Condition2']].apply(lambda x: ' '.join(x), axis=1)
    test['Conditions'] = test[['Condition1', 'Condition2']].apply(lambda x: ' '.join(x), axis=1)
    train['MonthYearSold'] = train[['MoSold', 'YrSold']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    test['MonthYearSold'] = test[['MoSold', 'YrSold']].astype(str).apply(lambda x: '_'.join(x), axis=1)

# Makee numerical and non-numerical subsets of data
    numerical = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Age', 'AgeRemod', 'Baths', 'BedroomAbvGr', 'BsmtBaths', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']
    non = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Conditions', 'Electrical', 'ExterCond', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MiscFeature', 'MonthYearSold', 'MSSubClass', 'MSZoning', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']

    train_num = train[numerical]
    train_num.reset_index(drop=True, inplace=True)
    train_non = train[non]
    train_non.reset_index(drop=True, inplace=True)
    test_num = test[numerical]
    test_num.reset_index(drop=True, inplace=True)
    test_non = test[non]
    test_non.reset_index(drop=True, inplace=True)
    train_non = train_non.replace(np.NaN, '')
    train_num = train_num.replace(np.NaN, -999)
    test_non = test_non.replace(np.NaN, '')
    test_num = test_num.replace(np.NaN, -999)

# Scale only numerical data
    train_test_num = pd.concat((train_num, test_num), ignore_index=True)
    train_test_num, scaler = scale_data( train_test_num )
    train_num, _ = scale_data( train_num, scaler )
    test_num, _ = scale_data( test_num, scaler )
    train_num = pd.DataFrame(train_num, columns=numerical)
    test_num = pd.DataFrame(test_num, columns=numerical)

# Create sparse matrices, first only from numerical data
    train_data = sparse_df_to_array(train_num)
    test_data = sparse_df_to_array(test_num)
    train_test_non = pd.concat((train_non, test_non), ignore_index=True)

# Convert individual categorical columns to sparse matrices, add to existing sparse matrices
    for i, col_name in enumerate(train_test_non.columns):
        CV.fit(train_test_non[col_name])
        train_sparse = CV.transform(train_non[col_name])
        test_sparse = CV.transform(test_non[col_name])
        train_data = hstack((train_data, train_sparse), format='csr')
        test_data = hstack((test_data, test_sparse), format='csr')

    print("\n Sparse Train Set Matrix Dimensions: %d x %d" % (train_data.shape[0], train_data.shape[1]))
    print("\n Sparse Test Set Matrix Dimensions: %d x %d\n" % (test_data.shape[0], test_data.shape[1]))
    timer(start_time)