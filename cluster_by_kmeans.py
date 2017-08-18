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
import json
from textwrap import wrap

def run(
    anomaly_group_by_state,
    interested_data_fields,
    algorithm_parameters,
    result_save_path,
    result_id,
):

    anomaly_group_by_state = util.make_state_data_same_length(anomaly_group_by_state)

    print 'cluster_by_kmeans'
    for state_no in anomaly_group_by_state:
        print 'processing state', state_no

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
        ax_raw_data.set_title("state %s data, anomalies are seperated by gray vertical lines"%(state_no,))
        ax_raw_data.set_xlabel('anomalies seperated by gray vertival lines')
        ax_raw_data.set_xticklabels([])

        cluster_result = {}

        silhouette_x = []
        silhouette_y = []
        print "n_clusters:", 
        for n_clusters in range(2, X.shape[0]):
            print n_clusters, 
            kmeans = KMeans(
                n_clusters=n_clusters, 
                n_jobs=-2).fit(X)

            cluster_labels = kmeans.labels_.tolist()
            metric_silhouette = silhouette_samples(X, cluster_labels)
            silhouette_x.append(n_clusters)
            silhouette_y.append(metric_silhouette)


            cluster_result[n_clusters] = {}
            cluster_result[n_clusters]['silhouette_score'] = metric_silhouette.mean() 
            cluster_result[n_clusters]['labels'] = cluster_labels
        print 'Done'



        ax_silhouette = fig.add_subplot(212)
        ax_silhouette.boxplot(silhouette_y, positions=silhouette_x)

        silhouette_mean = [np.array(i).mean() for i in silhouette_y]
        ax_silhouette.plot(silhouette_x, silhouette_mean, 'rs')
        ax_silhouette.set_title("\n".join(wrap("state %s result_id(\"%s\") silhouette score"%(state_no, result_id))))
        ax_silhouette.set_xlabel('number of clusters')

        output_path = os.path.join(result_save_path, 'state_%s'%state_no)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        subplot_amount = len(fig.get_axes())
        fig.set_size_inches(8,8*subplot_amount)
        fig.savefig(os.path.join(output_path, "state_%s_clustering_results.png"%state_no), format="png", bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

        json.dump(
            cluster_result,
            open(os.path.join(output_path, 'cluster_result.json'), 'w'),
            indent=4,
        )
