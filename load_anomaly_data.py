import os
import pandas as pd

def run(anomaly_data_folder_path, interested_data_fields):
    list_of_anomaly = []

    import re
    prog = re.compile('extracted_error_(\d+).csv')
    
    for f_1 in os.listdir(anomaly_data_folder_path):
        path_1 = os.path.join(anomaly_data_folder_path, f_1)
        if not os.path.isdir(path_1):
            continue
        for f_2 in os.listdir(path_1):
            m = prog.match(f_2)
            if not m:
                continue
            path_2 = os.path.join(path_1, f_2)
    
            trial_name = f_1
            anomaly_no = m.group(1)

            anomaly_id = "trial_(%s)_anomaly_(%s)"%(trial_name, anomaly_no)
            anomaly_df = pd.read_csv(path_2, sep=',')
            anomaly_df = anomaly_df[interested_data_fields].loc[anomaly_df['.tag'] != 0]
            states_in_df = anomaly_df['.tag'].unique().tolist()
            if len(states_in_df) != 1:
                print("%s is not of only one state: %s"%(anomaly_id, states_in_df)) 
                continue
            state = states_in_df[0]

            mat = anomaly_df.drop('.tag', axis=1).values

            list_of_anomaly.append({
                'anomaly_id': anomaly_id,
                'state_no': state,
                'data_matrix': mat,
            }) 

    return list_of_anomaly