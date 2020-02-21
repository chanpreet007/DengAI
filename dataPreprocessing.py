import pandas as pd
from math import floor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def add_lag(dataframe,
            columns,
            number_of_lag):

    for column in columns:  # for each feature (column)
        for lag in range(number_of_lag):  # for each time step (lag)
            dataframe[column + "_time - " + str(lag)] = dataframe[column].shift(lag)  # copy the previous value

    # remove the first number_of_lag rows
    dataframe.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    #change_dataframe.fillna(method='backfill', inplace=True)

    return dataframe

def add_lag_array(dataframe,
            columns,
            number_of_lag):

    for column in columns:  # for each feature (column)
        for lag in range(number_of_lag):  # for each time step (lag)
            dataframe[column + "_time - " + str(lag)] = dataframe[column].shift(lag)  # copy the previous value

    # remove the first number_of_lag rows
    dataframe.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    #change_dataframe.fillna(method='backfill', inplace=True)

    return dataframe


def preprocess_data(data_path,                                                          
                    labels_path=None,
                    lag_step_for_SanJuan=10,
                    lag_step_for_Iquitos=10):


    # load data and set index to city, year, weekofyear
    data_frame = pd.read_csv(data_path, index_col=[0, 1, 2])

   # data_frame = select_principal_components(data_frame,5)

    features = ['reanalysis_specific_humidity_g_per_kg',
                  'reanalysis_dew_point_temp_k',
                  'reanalysis_min_air_temp_k',
                   'station_min_temp_c']

    data_frame = data_frame[features]

    # fill missing values
    #dataframe.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    data_frame.fillna(method='ffill', inplace=True)


    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        data_frame = data_frame.join(labels)

    # separate san juan and iquitos
    sanJuan_data = data_frame.loc['sj']
    iquitos_data = data_frame.loc['iq']

    # add lag
    sanJuan_data = add_lag(sanJuan_data, features, lag_step_for_SanJuan)
    iquitos_data = add_lag(iquitos_data, features, lag_step_for_Iquitos)

    # fill navalues
    #sanJuan_data.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    #iquitos_data.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    sanJuan_data.fillna(method='backfill', inplace=True)
    iquitos_data.fillna(method='backfill', inplace=True)

    return sanJuan_data, iquitos_data

def preprocess_data_with_PCA(data_path,                    
                    labels_path=None,
                    lag_step_for_SanJuan=10,
                    lag_step_for_Iquitos=10):


    # load data and set index to city, year, weekofyear
    data_frame = pd.read_csv(data_path, index_col=[0, 1, 2])

   # data_frame = select_principal_components(data_frame,5)

    features = ['reanalysis_specific_humidity_g_per_kg',
                  'reanalysis_dew_point_temp_k',
                  'reanalysis_min_air_temp_k',
                   'station_min_temp_c']

    data_frame = data_frame[features]

    # fill missing values
    #data_frame.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    data_frame.fillna(method='ffill', inplace=True)


    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        data_frame = data_frame.join(labels)
    print(data_frame)
    # separate san juan and iquitos
    sanJuan_data = data_frame.loc['sj']
    iquitos_data = data_frame.loc['iq']
    
    # separate features from targets

    sanJuan_features = sanJuan_data.drop(["total_cases"], axis=1)
    sanJuan_targets = sanJuan_data['total_cases']

    iquitos_features = iquitos_data.drop('total_cases', axis=1)
    iquitos_targets = iquitos_data['total_cases']


    # apply PCA
    number_of_components = 10

        #for San Juan
    pca = PCA(n_components=number_of_components)
    pca.fit(sanJuan_features)
    sanJuan_features = pd.DataFrame(pca.transform(sanJuan_features),
                                columns=['PCA%i' % i for i in range(number_of_components)],
                                index=sanJuan_features.index)

    featuresNames_sanJuan = sanJuan_features.columns

        #for Iquitos
    pca = PCA(n_components=number_of_components)
    pca.fit(iquitos_features)
    iquitos_features  = pd.DataFrame(pca.transform(iquitos_features),
                                 columns=['PCA%i' % i for i in range(number_of_components)],
                                 index=iquitos_features.index)

    featuresNames_iquitos = iquitos_features.columns


    # add lag
    sanJuan_features = add_lag(sanJuan_features, featuresNames_sanJuan, lag_step_for_SanJuan)
    iquitos_features = add_lag(iquitos_features, featuresNames_iquitos, lag_step_for_Iquitos)

    # fill navalues
    #sanJuan_features.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    #iquitos_features.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

    sanJuan_features.fillna(method='backfill', inplace=True)
    iquitos_features.fillna(method='backfill', inplace=True)

    return sanJuan_features, sanJuan_targets, iquitos_features, iquitos_targets


def preprocess_data_with_PCA_for_test(data_path,
                             labels_path=None,
                             lag_step_for_SanJuan=10,
                             lag_step_for_Iquitos=10):
    # load data and set index to city, year, weekofyear
    data_frame = pd.read_csv(data_path, index_col=[0, 1, 2])

    # data_frame = select_principal_components(data_frame,5)

    features = ['reanalysis_specific_humidity_g_per_kg',
                  'reanalysis_dew_point_temp_k',
                  'reanalysis_min_air_temp_k',
                   'station_min_temp_c']

    data_frame = data_frame[features]

    # fill missing values
     #dataframe.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    data_frame.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        data_frame = data_frame.join(labels)

    # separate san juan and iquitos
    sanJuan_features = data_frame.loc['sj']
    iquitos_features = data_frame.loc['iq']


    # apply PCA
    number_of_components = 10

    # for San Juan
    pca = PCA(n_components=number_of_components)
    pca.fit(sanJuan_features)
    sanJuan_features = pd.DataFrame(pca.transform(sanJuan_features),
                                    columns=['PCA%i' % i for i in range(number_of_components)],
                                    index=sanJuan_features.index)

    featuresNames_sanJuan = sanJuan_features.columns

    # for Iquitos
    pca = PCA(n_components=number_of_components)
    pca.fit(iquitos_features)
    iquitos_features = pd.DataFrame(pca.transform(iquitos_features),
                                    columns=['PCA%i' % i for i in range(number_of_components)],
                                    index=iquitos_features.index)

    featuresNames_iquitos = iquitos_features.columns

    # add lag
    sanJuan_features = add_lag(sanJuan_features, featuresNames_sanJuan, lag_step_for_SanJuan)
    iquitos_features = add_lag(iquitos_features, featuresNames_iquitos, lag_step_for_Iquitos)

    # fill navalues
    sanJuan_features.fillna(method='backfill', inplace=True)
    iquitos_features.fillna(method='backfill', inplace=True)

    return sanJuan_features, iquitos_features

def split_time_series(X,
                      y,
                      test_set_ratio):

    number_of_examples = X.shape[0]
    last_index_in_train_set = floor((1 - test_set_ratio) * number_of_examples)

    X_train = X[:last_index_in_train_set]
    X_test = X[last_index_in_train_set:]

    y_train = y[:last_index_in_train_set]
    y_test = y[last_index_in_train_set:]

    return X_train, X_test, y_train, y_test
#def  preprocess_data_with_LDA(
