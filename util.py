def make_state_data_same_length(anomaly_group_by_state):
    for state_no in anomaly_group_by_state:
        lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
        min_len = min(lengths)
        anomaly_group_by_state[state_no]['list_of_mat'] = [\
            i[:min_len] for i in \
            anomaly_group_by_state[state_no]['list_of_mat']\
        ]
    return anomaly_group_by_state
