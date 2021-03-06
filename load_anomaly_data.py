import os
import pandas as pd
import ipdb
import util
from sklearn import preprocessing 
import numpy as np
from sklearn.decomposition import PCA
import ipdb

def preprocess_loaded_data(anomaly_group_by_state, data_preprocessing_config):
    if 'global_scaler' in data_preprocessing_config \
        and data_preprocessing_config['global_scaler']['turn_on']:
        feature_range = data_preprocessing_config['global_scaler']['parameters']['feature_range']
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)

        for state_no in anomaly_group_by_state:
            lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
            big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])
            for col_no in range(big_mat.shape[1]):
                vec = big_mat[:, col_no].reshape(-1, 1)
                big_mat[:, col_no] = min_max_scaler.fit_transform(vec).reshape(1, -1)
            for idx, i, j in util.iter_from_X_lengths(big_mat, lengths):
                mat = big_mat[i:j]
                anomaly_group_by_state[state_no]['list_of_mat'][idx] = mat


    if 'pca' in data_preprocessing_config \
        and data_preprocessing_config['pca']['turn_on']:

        for state_no in anomaly_group_by_state:
            lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
            big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])

            pca = PCA().fit(big_mat)
            big_mat = pca.transform(big_mat)

            for idx, i, j in util.iter_from_X_lengths(big_mat, lengths):
                mat = big_mat[i:j]
                anomaly_group_by_state[state_no]['list_of_mat'][idx] = mat

    return anomaly_group_by_state 

def run(
    anomaly_data_folder_path, 
    interested_data_fields,
    data_preprocessing_config,
):
    list_of_anomaly = []

    import re
    prog = re.compile('extracted_anomaly_(\d+).csv')
    
    for trial_name in os.listdir(anomaly_data_folder_path):
        trial_path = os.path.join(anomaly_data_folder_path, trial_name)
        if not os.path.isdir(trial_path):
            continue
        for anomaly_csv in os.listdir(trial_path):
            m = prog.match(anomaly_csv)
            if not m:
                continue
            anomlay_csv_path = os.path.join(trial_path, anomaly_csv)
    
            anomaly_no = m.group(1)

            anomaly_id = "trial_(%s)_anomaly_(%s)"%(trial_name, anomaly_no)
            raw_anomaly_df = pd.read_csv(anomlay_csv_path , sep=',')
            anomaly_df = raw_anomaly_df[interested_data_fields].loc[raw_anomaly_df['.tag'] != 0]
            states_in_df = anomaly_df['.tag'].unique().tolist()
            if len(states_in_df) != 1:
                print("%s is not of only one state: %s"%(anomaly_id, states_in_df)) 
                continue
            state = states_in_df[0]
    
            anomaly_df = anomaly_df.drop('.tag', axis=1)
            mat = anomaly_df.values

            list_of_anomaly.append({
                'anomaly_id': anomaly_id,
                'state_no': state,
                'data_matrix': mat,
                'raw_anomaly_df': raw_anomaly_df,
            }) 

    anomaly_group_by_state = {}
    for i in list_of_anomaly:
        state_no = i['state_no']
        if state_no not in anomaly_group_by_state:
            anomaly_group_by_state[state_no] = {
                'list_of_mat':[],
                'list_of_raw_df':[],
                'mat_owners':[],
            }
            
        anomaly_group_by_state[state_no]['list_of_raw_df'].append(i['raw_anomaly_df'])
        anomaly_group_by_state[state_no]['list_of_mat'].append(i['data_matrix'])
        anomaly_group_by_state[state_no]['mat_owners'].append(i['anomaly_id'])

    anomaly_group_by_state = preprocess_loaded_data(anomaly_group_by_state, data_preprocessing_config)

    return anomaly_group_by_state
