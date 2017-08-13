import matplotlib.pyplot as plt

def run(
    anomaly_group_by_state,
    interested_data_fields,
):

    for state_no in anomaly_group_by_state:
        fig, ax = plt.subplots(12, 1)
        for idx in range(len(ax)):
            ax[idx].set_title(interested_data_fields[idx])

        for mat in anomaly_group_by_state[state_no]['list_of_mat']:
            for col_no in range(mat.shape[1]):
                ax[col_no].plot(mat[:, col_no].reshape(1, -1).tolist()[0])
        fig.show()

    plt.show()
