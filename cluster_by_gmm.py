import numpy as np
import ipdb
import matplotlib.pyplot as plt
from sklearn import preprocessing, mixture
import ipdb
from matplotlib.pyplot import cm 
import matplotlib.mlab as mlab
import util


def project_to_gaussian_basis_space(mat):
    num_basis=31
    sigma = 0.05 
    means = np.arange(0,num_basis)/(num_basis-1.0)


    fig = plt.figure()
    ax_basis = fig.add_subplot(311)
    for i in range(num_basis):
        mu = means[i]  
        x = np.linspace(0, 1, 100)
        ax_basis.plot(x, mlab.normpdf(x, mu, sigma))

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

    
    ax_before = fig.add_subplot(312)
    ax_before.set_ylim((-1, 1))
    ax_after = fig.add_subplot(313)
    ax_after.set_ylim((-1, 1))
    projected_mat = np.zeros([num_basis, dim_amount])
    for col_no in range(dim_amount):
        ax_before.plot(
            np.linspace(0, 1, len_data),     
            mat[:, col_no].reshape(1, -1).tolist()[0],
        )
        W = np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), np.dot(Phi, mat[:, col_no]))
        approximated_y = np.dot(W, Phi)
        ax_after.plot(
            np.linspace(0, 1, len_data),     
            approximated_y,
        )

        mat[:, col_no] = approximated_y
        projected_mat[:, col_no] = W

    plt.close(fig)
    return mat, projected_mat 

def run(
    anomaly_group_by_state,
    interested_data_fields,
):

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    for state_no in anomaly_group_by_state:

        lengths = [i.shape[0] for i in anomaly_group_by_state[state_no]['list_of_mat']]
        big_mat = np.vstack(anomaly_group_by_state[state_no]['list_of_mat'])



        for col_no in range(big_mat.shape[1]):
            vec = big_mat[:, col_no].reshape(-1, 1)
            big_mat[:, col_no] = min_max_scaler.fit_transform(vec).reshape(1, -1)


        fig = plt.figure()
        bbox_extra_artists = []
        ax_raw_data = fig.add_subplot(311)
        ax_raw_data.set_ylim((-1, 1))
        ax_approximated_data = fig.add_subplot(312)
        ax_approximated_data.set_ylim((-1, 1))


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
        lgd = ax_raw_data.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)
        lgd = ax_approximated_data.legend(loc='center left', bbox_to_anchor=(1,0.5))
        bbox_extra_artists.append(lgd)


        bic_x = []
        bic_y = []

        X = np.matrix(X)
        sample_amount = X.shape[0]
        for n_components in range(1, sample_amount+1):
            gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full').fit(X)
            bic_x.append(n_components)
            bic_y.append(gmm.bic(X))

        ax_bic = fig.add_subplot(313)
        ax_bic.bar(bic_x, bic_y)
        ax_bic.set_title("state %s"%(state_no,))

    plt.show()

