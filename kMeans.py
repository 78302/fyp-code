# Implementation of K-means
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(100)

class KMeans:

    def __init__(self, k, data):
        """
        Run the K-means on given data with fixed number of clusters.
        :param k: Fixed number of clusters predefined ahead.
        :param data: Data for clustering.

        :return centers: Centers of the clusters.
        :return assigns: Correspondence between datapoint and cluster center with clustering errors.
        """
        self.k = k
        self.data = data
        self.centers = None
        self.assigns = None
        self.run_kmeans()

    def eu_distance(self, a, b):
        """
        Calculate the Euclidean Distance between two point
        :param a: Starting point
        :param b: End Point
        :return: Euclidean Distance
        """
        dist = np.sqrt(np.sum(np.square(a - b)))
        # np.linalg.norm(a - b)  # numpy function can replace the above
        return dist

    def assign_cluster(self, datapoint):
        """
        Assign the given datapoint to a cluster center
        :param datapoint: a data vector
        :return: c_index: The assigned cluster index
        :return: dist: The clustering error (L2 distance)
        """
        dists = np.array([self.eu_distance(c, datapoint) for c in self.centers])
        c_index = np.argmin(dists)
        dist = np.min(dists)
        return c_index, dist

    def run_kmeans(self):
        """
        Kernel steps of K-means
        """
        # Init centers
        self.centers = np.array(random.sample(list(self.data), self.k))  # randomly pick k data from data set as centers

        temp = None
        while True:
            # Assign data to clusters
            assigns = np.array([self.assign_cluster(datapoint) for datapoint in self.data])

            if (assigns == temp).all():  # satisfy the stop condition
                return

            temp = assigns  # store the old assigns

            # Updata centers
            for c_index in range(self.k):
                data_in_c = np.array([self.data[i] for i in range(self.data.shape[0]) if assigns[i][0] == c_index])
                self.centers[c_index] = np.mean(data_in_c, axis=0)

    def plot_result(self, num_classes=1):

        plt.figure(figsize=(8, 6))
        plt.title("Final centers in red")
        plt.scatter(self.data[:, 0], self.data[:, 1], marker='.', c=num_classes)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='r', marker='x', s=500)
        # plt.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1], c='k')
        plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    np.random.seed(1000)
    X, y = make_blobs(centers=4, n_samples=1000)  # generate samples

    kmeans = KMeans(4, X)
    kmeans.plot_result(y)