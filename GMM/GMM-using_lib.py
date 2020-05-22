
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold


#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def load_data(file_path):

    with open(file_path, 'r') as f:

        data = ';'.join(f.readlines())
        data = np.mat(data)
        # print(data)
    return data
    pass

class sklearn_gmm:

    def __init__(self):

        self.colors = ['navy', 'turquoise', 'darkorange']

    def make_ellipses(self, gmm, ax):

        for n, color in enumerate(self.colors):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                      180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect('equal', 'datalim')

    def runGMM(self, data, k):

        n_classes = k

        estimators = {cov_type: GaussianMixture(n_components=n_classes,
                      covariance_type=cov_type, max_iter=20, random_state=0)
                      for cov_type in ['spherical', 'diag', 'tied', 'full']}

        num_estimators = len(estimators)
        plt.figure(figsize=(3 * num_estimators // 2, 6))
        plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                            left=.01, right=.99)



        for index, (name, estimator) in enumerate(estimators.items()):

            estimator.fit(data)

            h = plt.subplot(2, num_estimators // 2, index + 1)
            self.make_ellipses(estimator, h)

            result = estimator.predict(data)
            # print(result)

            prob = estimator.predict_proba(data)
            # print(prob)


            plot_data = [[] for i in range(k)]
            for i, d in enumerate(data):
                plot_data[result[i]].append([d[0,0], d[0,1]])


            for n, color in enumerate(self.colors):

                p_data = np.mat(plot_data[n])
                # print(np.array(p_data[:,0]).reshape(-1))
                # plt.plot(p_data[:, 0], p_data[:, 1], 'ob', color=color)
                plt.scatter(np.array(p_data[:,0]).reshape(-1), np.array(p_data[:,1]).reshape(-1), s = 0.8, color=color)

            plt.title(name)

        plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


        plt.show()


if __name__ == '__main__':
    
    data = load_data('clusters.txt')
    # gmm = GMM(n_components=3).fit(data)
    gmm = sklearn_gmm()
    gmm.runGMM(data, 3)





