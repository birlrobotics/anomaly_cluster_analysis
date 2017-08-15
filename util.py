import numpy as np

def make_state_data_same_length(anomaly_group_by_state):
    for state_no in anomaly_group_by_state:
        lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
        min_len = min(lengths)
        anomaly_group_by_state[state_no]['list_of_mat'] = [\
            i[:min_len] for i in \
            anomaly_group_by_state[state_no]['list_of_mat']\
        ]
    return anomaly_group_by_state

def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield i, start[i], end[i]
