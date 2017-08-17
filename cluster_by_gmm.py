import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn import mixture
import ipdb
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples
from matplotlib.pyplot import cm 
import matplotlib.mlab as mlab
import util
import os
import json
from textwrap import wrap


def project_to_gaussian_basis_space(mat):
    num_basis=31
    sigma = 0.05 
    means = np.arange(0,num_basis)/(num_basis-1.0)

    len_data = mat.shape[0]
    dim_amount = mat.shape[1]
    x = np.linspace(0.0, 1.0, len_data)     # the time stamp
    Phi = np.exp(\
        -.5*(\
            np.array(\
                map(\
                    lambda arg: arg-means,\
                    np.tile(x, (num_basis, 1)).T\
                )\
            ).T**2 \
        / (sigma**2))\
    )

    
    projected_mat = np.zeros([num_basis, dim_amount])
    for col_no in range(dim_amount):
        W = np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), np.dot(Phi, mat[:, col_no]))
        approximated_y = np.dot(W, Phi)

        mat[:, col_no] = approximated_y
        projected_mat[:, col_no] = W

    return mat, projected_mat 

def run(
    anomaly_group_by_state,
    interested_data_fields,
    algorithm_parameters,
    result_save_path,
    result_id,
):


    for state_no in anomaly_group_by_state:

        lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
        big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])


        fig = plt.figure()
        fig.suptitle(result_save_path)
        bbox_extra_artists = []
        ax_raw_data = fig.add_subplot(311)
        ax_approximated_data = fig.add_subplot(312)


        dim_color = {}
        dim_amount = big_mat.shape[1]
        color=iter(cm.rainbow(np.linspace(0, 1, dim_amount)))
        for dim_no in range(dim_amount):
            dim_color[dim_no] = color.next()

        X = []
        for idx, i, j in util.iter_from_X_lengths(big_mat, lengths):
            mat = big_mat[i:j]

            for col_no in range(dim_amount):
                if idx == 0:
                    label = interested_data_fields[col_no]
                else:
                    label = None

                ax_raw_data.plot(
                    range(i, j),
                    mat[:, col_no].reshape(1, -1).tolist()[0],
                    label=label,
                    color=dim_color[col_no],
                )
            ax_raw_data.axvline(x=i, color='gray')

            approximated_mat, projected_mat = project_to_gaussian_basis_space(mat.copy())

            for col_no in range(dim_amount):
                if idx == 0:
                    label = interested_data_fields[col_no]
                else:
                    label = None

                ax_approximated_data.plot(
                    range(i, j),
                    approximated_mat[:, col_no].reshape(1, -1).tolist()[0],
                    label=label,
                    color=dim_color[col_no],
                )
            ax_approximated_data.axvline(x=i, color='gray')

            X.append(projected_mat.flatten().tolist())
        ax_raw_data.set_xticklabels([])
        ax_raw_data.set_title("state %s data, anomalies are seperated by gray vertical lines"%(state_no,))
        ax_raw_data.set_xlabel('anomalies seperated by gray vertival lines')
        lgd = ax_raw_data.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)

        ax_approximated_data.set_title("state %s data approximated by n gaussian basis functions, anomalies are seperated by gray vertical lines"%(state_no,))
        ax_approximated_data.set_xlabel('anomalies seperated by gray vertival lines')
        ax_approximated_data.set_xticklabels([])
        bbox_extra_artists.append(lgd)

        cluster_result = {}

        silhouette_x = []
        silhouette_y = []

        X = np.matrix(X)
        sample_amount = X.shape[0]
        for n_clusters in range(2, sample_amount):
            print "n_clusters:", n_clusters
            gmm = mixture.GaussianMixture(n_components=n_clusters,
                                      covariance_type='full').fit(X)

            cluster_labels = gmm.predict(X).tolist() 
            metric_silhouette = silhouette_samples(X, cluster_labels)
            silhouette_x.append(n_clusters)
            silhouette_y.append(metric_silhouette)


            cluster_result[n_clusters] = {}
            cluster_result[n_clusters]['silhouette_score'] = metric_silhouette.mean() 
            cluster_result[n_clusters]['labels'] = cluster_labels


        ax_silhouette = fig.add_subplot(313)
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
