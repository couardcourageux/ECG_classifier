import pandas as pd
import numpy as np
from sklearn import preprocessing





def create_x_y(df):
    # split the x's and y's
    x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    
    #encode and fit the labels
    encoder = preprocessing.LabelEncoder()
    y_transformed = encoder.fit_transform(y)
    encoder = preprocessing.OneHotEncoder()
    y_transformed = encoder.fit_transform(y_transformed.reshape(-1, 1))
    y_transformed = y_transformed.toarray()
    
    #fit the X's
    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_fitted = max_abs_scaler.fit_transform(x)
    
    #reshape the inner arrays of x_fitted
    print(x_fitted.shape)
    x_as_list = x_fitted.tolist()
    x_to_fitted = [np.reshape(x, (-1, 1)) for x in x_as_list]
    
    x_fitted = np.array(x_to_fitted)
    print(x_fitted.shape)
    
    
    return x_fitted, y_transformed


def get_train_test_fitted(df_train, df_test):
    # on merge les dataframe en vue d'homogénéiser la normalisation des données
    lg_train = df_train.shape[0]

    df_test = df_test.reset_index(drop=True)
    train_test_df = pd.concat([df_train, df_test], axis=0)
    
    # on fit les x et y sur l'ensemble du dataframe
    x_fitted, y_fitted = create_x_y(train_test_df)
    
    x_train, y_train = x_fitted[:lg_train], y_fitted[:lg_train]
    x_test, y_test = x_fitted[lg_train:], y_fitted[lg_train:]
    
    
    
    return x_train, y_train, x_test, y_test
    