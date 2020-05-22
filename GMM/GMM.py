import numpy as np

import imageio

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def load(txt_path):
    '''
    load data and turn it into matrix format
    :param txt_path:
    :return:
    '''
    with open(txt_path, 'r') as f:
        content = f.readlines()
    X = []
    for each_line in content:
        x = each_line.replace('\n', '')
        x0 = float(x.split(',')[0])
        x1 = float(x.split(',')[1])
        x = [x0, x1]
        X.append(x)
    X_M = np.array(X)
    return X_M


def random_num(k):
    generate_list = [np.random.rand() for i in range(k)]
    add_up = sum(generate_list)
    generate_list = [i / add_up for i in generate_list]
    return generate_list


def initialize_param(data, k):
    '''
    initilize the gamma
    :param data:
    :param k:
    :return:
    '''
    clusters = []
    sample_num = data.shape[0]  # 150
    feature_num = data.shape[1]  # 2

    gamma = np.zeros((sample_num, k))  # (150,3)
    # random generate 150 numbers and they add up to 1
    for i in range(sample_num):
        gamma_nk = random_num(k)
        gamma[i] = np.array(gamma_nk)

    mu_k = np.zeros(feature_num)  # (3,2)
    cov_k = []
    pi_k = 0  # each cluster has the same probability

    gamma_nk = gamma

    for i in range(k):
        clusters.append({
            'pi_k': pi_k,
            'mu_k': mu_k,
            'cov_k': cov_k,
            'gamma_nk': gamma[:, i]
        })

    return clusters


def update_gaussain_para(X, clusters):
    '''
    update mu cov and pi
    :param X:
    :param clusters:
    :return:
    '''
    N = X.shape[0]
    D = X.shape[1]
    for cluster in clusters:
        cov_k = np.zeros((D, D))  # 2 x 2
        gamma_nk = np.array(cluster['gamma_nk']).reshape(1, -1)
        # gamma_nk  (1,150)   ri1
        N_k = np.sum(gamma_nk)
        # update pi
        pi_k = N_k / N
        # update mu
        mu_k = gamma_nk.dot(X) / N_k
        # update cov
        for j in range(N):
            diff = (X[j] - mu_k).reshape(-1, 1)  # (2,1)
            cov_k += gamma_nk[0][j] * diff.dot(diff.T)
        cov_k /= N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k

    return clusters


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5)
                       * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def update_gamma(X, clusters):
    '''
    update gamma
    :param X:
    :param clusters:
    :return:
    '''
    totals = np.zeros((X.shape[0], 1))

    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']

        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)

        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]

        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals

    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']

    return clusters




# here I search the Internet for help , to find how to plot the graph

def get_likelihood(X, clusters):
    likelihood = []
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs):
    clusters = initialize_param(X, n_clusters)
    likelihoods = np.zeros((n_epochs,))
    scores = np.zeros((X.shape[0], n_clusters))
    history = []

    for i in range(n_epochs):
        clusters_snapshot = []

        # This is just for our later use in the graphs
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })

        history.append(clusters_snapshot)

        update_gaussain_para(X, clusters)
        update_gamma(X, clusters)

        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood

        # print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)

    return clusters, likelihoods, scores, sample_likelihoods, history


def create_cluster_animation(X, history, scores):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colorset = ['blue', 'red', 'black']
    images = []

    for j, clusters in enumerate(history):

        idx = 0
        if j % 3 != 0 or j == 0:
            continue

        plt.cla()
        for cluster in clusters:
            mu = cluster['mu_k'].reshape(-1, )
            cov = cluster['cov_k']

            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
            theta = np.arctan2(vy, vx)

            color = colors.to_rgba(colorset[idx])

            for cov_factor in range(1, 4):
                ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                              height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta), linewidth=2)
                ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                ax.add_artist(ell)

            ax.scatter(mu[0], mu[1], c=colorset[idx], s=1000, marker='+')
            idx += 1

        for i in range(X.shape[0]):
            ax.scatter(X[i, 0], X[i, 1], c=colorset[np.argmax(scores[i])], marker='o')

        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)

    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave('./gmm.gif', images, fps=1)


if __name__ == '__main__':

    # main part of GMM
    k = 3
    txt_path = 'clusters.txt'  # data stored in the data directory
    data = load(txt_path)
    clusters = initialize_param(data, k)
    clusters = update_gaussain_para(data, clusters)
    former_gamma_list = [list(i['gamma_nk']) for i in clusters]

    count = 0
    while True:
        # update gaussain_para
        update_gaussain_para(data, clusters)
        # update gamma
        update_gamma(data, clusters)
        new_gamma_list = [list(i['gamma_nk']) for i in clusters]
        if former_gamma_list == new_gamma_list:
            break
        else:
            former_gamma_list = new_gamma_list
        count += 1
        if count > 5000:
            break
    print(clusters) # after convergence  , the result of cluster


    # plot the graph  restart the GMM and visualize the process --- Searching from the Internet
    n_clusters = 3
    n_epochs = 100
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(data, n_clusters, n_epochs)
    create_cluster_animation(data, history, scores)
    # save the gif into the fold
