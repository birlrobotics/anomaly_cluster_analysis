import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score, silhouette_samples

def run(
    anomaly_group_by_state,
):

    for state_no in anomaly_group_by_state:
        print '-'*20
        print state_no
        X = []
        for mat in anomaly_group_by_state[state_no]['list_of_mat']:
            X.append(mat.flatten().tolist())

        silhouette_x = []
        silhouette_y = []
        X = np.matrix(X)
        for n_clusters in range(2, X.shape[0]):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                verbose=1,
                n_jobs=-2).fit(X)

            metric_silhouette = silhouette_samples(X, kmeans.labels_)
            #metric_calinski_harabaz_score = calinski_harabaz_score(X, kmeans.labels_)
            silhouette_x.append(n_clusters)
            silhouette_y.append(metric_silhouette)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(silhouette_y, positions=silhouette_x)
        silhouette_mean = [np.array(i).mean() for i in silhouette_y]
        ax.plot(silhouette_x, silhouette_mean, 'rs')
        ax.set_title("state %s"%(state_no,))
        fig.show()

    plt.show()

