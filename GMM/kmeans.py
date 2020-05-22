import random
import numpy as np
import matplotlib.pyplot as plt

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)


def load(txt_path):
    '''
    load txt data and turn it into matrix format
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


def initialize_centroid(data, k):
    '''
    initialize the centroid for k's cluster
    :param data:
    :param k:
    :return:
    '''
    data_list = list(data)
    centroids_list = random.sample(data_list, k) # pick different samples , here data_list has to be list type
    centroids = np.array(centroids_list)
    return centroids


def calculate_dist(xi, centroids):
    '''
    calculate the distance between data point and each centroid
    :param xi:
    :param centroids:
    :return:
    '''
    distances = np.sqrt(np.sum((xi - centroids) ** 2, axis=1))
    return distances


def assign_to_clsuters(data, centroids):
    '''
    assign each data point to its nearest centroid
    :param data:
    :param centroids:
    :return:
    '''
    cluster_list = [[] for i in range(len(centroids))]
    for point in data:
        distances = calculate_dist(point, centroids)  # (k, )
        min_dis = min(distances)
        assign_cluster = list(distances).index(min_dis)
        cluster_list[assign_cluster].append(point)

    return cluster_list


def recompute_centroid(cluster_list):
    '''
    for each cluster, recompute its centroid
    :param cluster_list:
    :return:
    '''
    new_centroids_list = []

    for cluster in cluster_list:
        data_points_list = cluster[:]
        data_points = np.array(data_points_list)
        mean = np.mean(data_points, axis=0)

        new_centroids_list.append(mean)

    new_centroids = np.array(new_centroids_list)

    return new_centroids


if __name__ == '__main__':

    k = 3
    txt_path = 'clusters.txt' # data stored in the data directory
    data = load(txt_path)

    initial_centroids = initialize_centroid(data, k)
    cluster_list = assign_to_clsuters(data, initial_centroids)

    count = 0
    former_centroids = initial_centroids
    new_centroids = []

    while True:

        cluster_list = assign_to_clsuters(data, former_centroids)
        new_centroids = recompute_centroid(cluster_list)

        if (former_centroids == new_centroids).all():
            break
        else:
            former_centroids = new_centroids
        count += 1
        if count > 10000:
            break

    print('After clustering, the centroids are :')
    print(new_centroids)

    # here plot the result to show the effect
    mark = ['or', 'ob', 'og', ]
    for i, cluster in enumerate(cluster_list):
        for point in cluster:
            plt.plot(point[0], point[1], mark[i], markersize=5)

    for i in range(k):
        plt.plot(new_centroids[i][0], new_centroids[i][1], mark[i], markersize=12)

    plt.show()