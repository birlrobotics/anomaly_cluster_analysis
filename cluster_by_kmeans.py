import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
from sklearn import preprocessing
import ipdb
from matplotlib.pyplot import cm 

def run(
    anomaly_group_by_state,
    interested_data_fields,
):

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    for state_no in anomaly_group_by_state:

        sample_length = anomaly_group_by_state[state_no]['list_of_mat'][0].shape[0]
        big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])



        for col_no in range(big_mat.shape[1]):
            vec = big_mat[:, col_no].reshape(1, -1).tolist()[0]
            big_mat[:, col_no] = min_max_scaler.fit_transform(vec)

        fig = plt.figure()
        bbox_extra_artists = []
        ax_raw_data = fig.add_subplot(211)


        dim_color = {}
        dim_amount = big_mat.shape[1]
        color=iter(cm.rainbow(np.linspace(0, 1, dim_amount)))
        for dim_no in range(dim_amount):
            dim_color[dim_no] = color.next()

        X = []
        for idx in range(0, big_mat.shape[0], sample_length):
            mat = big_mat[idx:idx+sample_length]
            X.append(mat.flatten().tolist())

            for col_no in range(mat.shape[1]):
                if idx == 0:
                    label = interested_data_fields[col_no]
                else:
                    label = None
                ax_raw_data.plot(
                    range(idx, idx+sample_length),
                    mat[:, col_no].reshape(1, -1).tolist()[0],
                    label=label,
                    color=dim_color[col_no],
                )
            ax_raw_data.axvline(x=idx, color='gray')
        lgd = ax_raw_data.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)



        silhouette_x = []
        silhouette_y = []
        X = np.matrix(X)
        for n_clusters in range(2, X.shape[0]):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                verbose=1,
                n_jobs=-2).fit(X)

            metric_silhouette = silhouette_samples(X, kmeans.labels_)
            silhouette_x.append(n_clusters)
            silhouette_y.append(metric_silhouette)
        ax_silhouette = fig.add_subplot(212)
        ax_silhouette.boxplot(silhouette_y, positions=silhouette_x)
        silhouette_mean = [np.array(i).mean() for i in silhouette_y]
        ax_silhouette.plot(silhouette_x, silhouette_mean, 'rs')
        ax_silhouette.set_title("state %s"%(state_no,))

    plt.show()

