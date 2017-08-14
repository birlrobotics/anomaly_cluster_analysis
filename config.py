import os 

# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'twist',
    'wrench',
]
from data_fields_config_store import data_fields_store

config_by_user = {
    'data_type_chosen': data_type_options[:],
    'cluster_algorithm': {
        'name': 'kmeans',
        'parameters': {

        }
    },
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

exec '\n'.join("%s=%r"%i for i in config_by_user.items())
