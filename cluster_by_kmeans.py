import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
import ipdb
from matplotlib.pyplot import cm 
import util
from sklearn.decomposition import PCA
import os

def run(
    anomaly_group_by_state,
    interested_data_fields,
    algorithm_parameters,
    figure_save_path,
):

    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)


    anomaly_group_by_state = util.make_state_data_same_length(anomaly_group_by_state)

    for state_no in anomaly_group_by_state:

        sample_length = anomaly_group_by_state[state_no]['list_of_mat'][0].shape[0]
        big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])

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
        X = np.matrix(X)

        lgd = ax_raw_data.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)
        ax_raw_data.set_title("state %s data, trials are seperated by gray vertical lines"%(state_no,))
        ax_raw_data.set_xlabel('trails seperated by gray vertival lines')
        ax_raw_data.set_xticklabels([])

        silhouette_x = []
        silhouette_y = []
        for n_clusters in range(2, X.shape[0]):
            print X.shape
            kmeans = KMeans(
                n_clusters=n_clusters, 
                n_jobs=-2).fit(X)

            metric_silhouette = silhouette_samples(X, kmeans.labels_)
            silhouette_x.append(n_clusters)
            silhouette_y.append(metric_silhouette)

        ax_silhouette = fig.add_subplot(212)
        ax_silhouette.boxplot(silhouette_y, positions=silhouette_x)

        silhouette_mean = [np.array(i).mean() for i in silhouette_y]
        ax_silhouette.plot(silhouette_x, silhouette_mean, 'rs')

        ax_silhouette.set_title("state %s kmeans silhouette score"%(state_no,))
        ax_silhouette.set_xlabel('number of clusters')

        fig.savefig(os.path.join(figure_save_path, "state_%s_clustering_results.png"%state_no), format="png", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight', dpi=900)
