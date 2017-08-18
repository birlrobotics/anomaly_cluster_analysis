import os 
import ipdb
import util

# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'twist',
    'wrench',
]
from data_fields_config_store import data_fields_store

cluster_algorithm_options = [
    {
        'name': 'kmeans',
        'parameters': {
        }
    },
    {
        'name': 'gmm',
        'parameters': {
        }
    },
]

config_by_user = {
    'data_type_chosen': data_type_options[1:],
    'cluster_algorithm': cluster_algorithm_options[1], 
    'data_preprocessing_config': {
        'global_scaler': {
            'turn_on': True,
            'parameters': {
                'feature_range': [-1, 1],
            },
        },
        'pca': {
            'turn_on': True,
            'parameters': {
            },
        },
    },
    'data_folder_path': '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
}


config_to_traverse = [
    {
        "config_to_assign": 'config.config_by_user["data_type_chosen"]',
        'values_to_try': util.powerset_without_empty_set(data_type_options),
    },
    {
        "config_to_assign": 'config.config_by_user["cluster_algorithm"]',
        'values_to_try': cluster_algorithm_options,
    },
    {
        "config_to_assign": 'config.config_by_user["data_preprocessing_config"]["global_scaler"]["turn_on"]',
        'values_to_try': [False, True],
    },
    {
        "config_to_assign": 'config.config_by_user["data_preprocessing_config"]["pca"]["turn_on"]',
        'values_to_try': [False, True],
    },
]


def generate_auto_config():
    global interested_data_fields
    global anomaly_data_path
    global analysis_folder_path 
    global result_save_path
    global result_id 

    if type(config_by_user['data_type_chosen']) is not list:
        config_by_user['data_type_chosen'] = [config_by_user['data_type_chosen']]
    from data_fields_config_store import data_fields_store
    interested_data_fields = []
    for data_type in config_by_user['data_type_chosen']:
        interested_data_fields += data_fields_store[data_type]
    interested_data_fields.append('.tag')

    anomaly_data_path = os.path.join(config_by_user['data_folder_path'], "has_anomaly")
    analysis_folder_path = os.path.join(config_by_user['data_folder_path'], "anomaly_cluster_analysis")
    result_save_path = os.path.join(
        analysis_folder_path,
        'datatype_'+str(config_by_user['data_type_chosen']), 
        'algorithm_'+config_by_user['cluster_algorithm']['name'],
        'config_'+'processing_config_'+str(config_by_user['data_preprocessing_config']),
    )
    tmp_str = str(result_save_path)
    import re
    tmp_idx = re.search(r'anomaly_cluster_analysis', tmp_str).start()
    result_id = tmp_str[tmp_idx+len('anomaly_cluster_analysis'):]

    for name, value in config_by_user.items():
        globals()[name] = value

generate_auto_config()
