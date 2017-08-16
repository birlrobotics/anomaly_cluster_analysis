import os 

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
    'data_type_chosen': data_type_options[:],
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
    'cluster_algorithm': cluster_algorithm_options[0], 
    'data_folder_path': '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
}


if type(config_by_user['data_type_chosen']) is not list:
    config_by_user['data_type_chosen'] = [config_by_user['data_type_chosen']]
from data_fields_config_store import data_fields_store
interested_data_fields = []
for data_type in config_by_user['data_type_chosen']:
    interested_data_fields += data_fields_store[data_type]
interested_data_fields.append('.tag')

anomaly_data_path = os.path.join(config_by_user['data_folder_path'], "has_anomaly")
figure_save_path = os.path.join(
    config_by_user['data_folder_path'], 
    "figure", 
    "anomaly_cluster_analysis", 
    'datatype_'+str(config_by_user['data_type_chosen']), 
    'algorithm_'+config_by_user['cluster_algorithm']['name'],
    'config_'+'processing_config_'+str(config_by_user['data_preprocessing_config']),
)

exec '\n'.join("%s=%r"%i for i in config_by_user.items())
