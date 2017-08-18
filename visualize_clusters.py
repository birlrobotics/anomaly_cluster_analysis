import os
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import numpy as np
from textwrap import wrap
import random

def run(
    anomaly_group_by_state,
    df_group_by_foldername,
    result_save_path,
    result_id,
):

    print 'visualize_clusters'
    for state_no in anomaly_group_by_state:
        print 'processing state', state_no
        output_path = os.path.join(result_save_path, 'state_%s'%state_no)
        cluster_result = json.load(
            open(os.path.join(output_path, 'cluster_result.json'), 'r'),
        )
        list_of_anomaly_df = anomaly_group_by_state[state_no]['list_of_raw_df']
        list_of_anomaly_id = anomaly_group_by_state[state_no]['mat_owners']

        best_cluster = None
        best_score = 0
        for n_clusters in cluster_result:
            silhouette_score = cluster_result[n_clusters]['silhouette_score']
            if silhouette_score > best_score:
                best_score = silhouette_score
                best_cluster = n_clusters

        print "best_cluster:", best_cluster, 

        n_clusters = best_cluster

        silhouette_score = cluster_result[n_clusters]['silhouette_score']
        labels = cluster_result[n_clusters]['labels']

        df_group_by_label = {}
        for idx in range(len(labels)):
            label = labels[idx]
            df = list_of_anomaly_df[idx]
            id = list_of_anomaly_id[idx]
            if label not in df_group_by_label:
                df_group_by_label[label] = {
                    'list_of_df':[],
                    'list_of_id':[],
                }
            df_group_by_label[label]['list_of_df'].append(df)
            df_group_by_label[label]['list_of_id'].append(id)

        color=iter(cm.rainbow(np.linspace(0, 1, n_clusters)))

        fig = plt.figure()
        bbox_extra_artists = []
        pos_plot = fig.add_subplot(111, projection='3d')

        for f, raw_df in df_group_by_foldername.iteritems():
            df = raw_df
            pos_plot.plot(
                df['.endpoint_state.pose.position.x'].tolist(), 
                df['.endpoint_state.pose.position.y'].tolist(), 
                df['.endpoint_state.pose.position.z'].tolist(), 
                color='gray',
                alpha=0.1,
            )

        for label in df_group_by_label:
            label_color = next(color)
            for idx in range(len(df_group_by_label[label]['list_of_df'])):
                df = df_group_by_label[label]['list_of_df'][idx]
                id = df_group_by_label[label]['list_of_id'][idx]

                pos_mat = np.matrix([\
                    df['.endpoint_state.pose.position.x'].tolist(),\
                    df['.endpoint_state.pose.position.y'].tolist(),\
                    df['.endpoint_state.pose.position.z'].tolist(),\
                ])

                vec_mat = pos_mat[:, 1:]-pos_mat[:, :-1] 
                pos_mat = pos_mat[:, :-1]

                traj_length = pos_mat.shape[1]
                alpha_step = 10 
                length_step = traj_length/alpha_step
                for idx in range(1, alpha_step+1):
                    pos_plot.quiver(
                        pos_mat[0, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        pos_mat[1, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        pos_mat[2, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        vec_mat[0, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        vec_mat[1, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        vec_mat[2, idx*length_step:(idx+1)*length_step].tolist()[0], 
                        color=label_color,
                        alpha=0.05+0.95*idx/alpha_step,
                    )
        pos_plot.set_title("\n".join(wrap("state %s cluster %s result_id(\"%s\") silhouette_score %s"%(state_no, n_clusters, result_id, silhouette_score))))
        lgd = pos_plot.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)

        output_path = os.path.join(result_save_path, 'state_%s'%state_no)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        subplot_amount = len(fig.get_axes())
        fig.set_size_inches(8,8*subplot_amount)
        fig.savefig(os.path.join(output_path, "state %s cluster %s.png"%(state_no, n_clusters)), format="png", additional_artists=bbox_extra_artists, bbox_inches='tight')

        plt.close(fig)
        print 'Done'
