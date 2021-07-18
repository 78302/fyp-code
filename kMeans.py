# Implementation of K-means
import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob
import matplotlib.pyplot as plt
from functions import assign_cluster, pretrain_representations
import random

import argparse
# Deal with the use input parameters
# ideal input: name, epoch, K, pretrain_model_path, hidden-size, train-scp-path, dev-scp-path, layers
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=10)
parser.add_argument('--literable', '-l', help='wheter use multiple Ks to find the best one, not required', type=int,  default=1)
parser.add_argument('--cluster_number', '-k', help='The largest number of clusters, not required', type=int,  default=50)
parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)
parser.add_argument('--scp_file', '-s', help='Path of the input training utterance scp file, not required', default='./data/raw_fbank_train_si284.1.scp')
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
EPOCH = args.epoch
L = args.literable
K = args.cluster_number
PRETRAIN_PATH = args.model_path
SCP_FILE = args.scp_file

random.seed(100)
total_loss = []

import time
start_time = time.time()
tmp = start_time

if L:
    for k in range(1, K+1):
        start = True
        epochs = 0
        temp = None
        checkpoint = tmp
        for e in range(EPOCH):
            epoch_error = 0

            # Judge whether it can be free from the loop
            # use if condition
            # --------------
            # #################

            # Read the SCP file
            with open(SCP_FILE, 'rb') as scp_file:
                # mlp use '../remote/data/wsj/fbank/' replace '/data/'
                lines = scp_file.readlines()
                for line in lines:

                    tempt = str(line).split()[1]
                    file_loc = tempt.split(':')[0][18:]  # mlp keep [18:]
                    pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

                    # Read the ark file to get utterance
                    with open('../remote/data' + file_loc, 'rb') as ark_file:
                        # use '../remote/data' + file_loc replace './data/' + file_loc
                        ark_file.seek(int(pointer))
                        utt_mat = kaldiark.parse_feat_matrix(ark_file)

                        # Use model to get representations
                        if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
                            utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)

                        # Init centers: randomly pick k data from data set as centers
                        if start:
                            centers = np.array(random.sample(list(utt_mat), k))  # k=4
                            start = False
                            print(centers.shape)

                        # Assign data to clusters
                        assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])

                        # Update centers
                        for c_index in range(k):  # k=4
                            data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
                            # print(data_in_c.shape)
                            if data_in_c.shape[0] > 0:  # some clusters may not have assigned data points
                                centers[c_index] = np.mean(data_in_c, axis=0)

                        # Calculate the clustering loss
                        assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
                        epoch_error += np.sum(assigns, axis=0)[1]

            temp = epoch_error  # store the old assigns

            end = time.time()
            print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), epoch_error, (end-tmp)))
            tmp = end
            epochs += 1
        total_loss.append(epoch_error)
        print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, epoch_error, (end-checkpoint)))
    print("The best number of clusters is {:d}, the total trail costs {:0.7f} seconds".format((total_loss.index(min(total_loss))+1), (end-start_time)))

    # Draw the loss trend figure
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y = total_loss
    x = np.arange(1, K+1)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x,y,'r--',label='Cluster loss trend')

    ax.set_title('Clustering Loss with various cluster numbers',fontsize=18)
    ax.set_xlabel('Number of cluster', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
    ax.set_ylabel('Final clustering Loss', fontsize='x-large',fontstyle='oblique')
    ax.legend()

    plt.savefig("./kmeans/graph/{:s}_cluster_loss.pdf".format(NAME))

else:
    start = True
    epochs = 0
    temp = None
    checkpoint = tmp
    k=K
    for e in range(EPOCH):
        epoch_error = 0

        # Judge whether it can be free from the loop
        # use if condition
        # --------------
        # #################

        # Read the SCP file
        with open(SCP_FILE, 'rb') as scp_file:
            # mlp use '../remote/data/wsj/fbank/' replace '/data/'
            lines = scp_file.readlines()
            for line in lines:

                tempt = str(line).split()[1]
                file_loc = tempt.split(':')[0][18:]  # mlp keep [18:]
                pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

                # Read the ark file to get utterance
                with open('../remote/data' + file_loc, 'rb') as ark_file:
                    # use '../remote/data' + file_loc replace './data/' + file_loc
                    ark_file.seek(int(pointer))
                    utt_mat = kaldiark.parse_feat_matrix(ark_file)

                    # Use model to get representations
                    if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
                        utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)

                    # Init centers: randomly pick k data from data set as centers
                    if start:
                        centers = np.array(random.sample(list(utt_mat), k))  # k=4
                        start = False
                        print(centers.shape)

                    # Assign data to clusters
                    assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])

                    # Update centers
                    for c_index in range(k):  # k=4
                        data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
                        # print(data_in_c.shape)
                        if data_in_c.shape[0] > 0:  # some clusters may not have assigned data points
                            centers[c_index] = np.mean(data_in_c, axis=0)

                    # Calculate the clustering loss
                    assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
                    epoch_error += np.sum(assigns, axis=0)[1]

        temp = epoch_error  # store the old assigns

        end = time.time()
        print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), epoch_error, (end-tmp)))
        tmp = end
        epochs += 1
    total_loss.append(epoch_error)
    print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, epoch_error, (end-checkpoint)))



# for k in range(1, K+1):
#     start = True
#     epochs = 0
#     temp = None
#     checkpoint = tmp
#     for e in range(EPOCH):
#         epoch_error = 0
#
#         # Judge whether it can be free from the loop
#         # use if condition
#         # --------------
#         # #################
#
#         # Read the SCP file
#         with open(SCP_FILE, 'rb') as scp_file:
#             # mlp use '../remote/data/wsj/fbank/' replace '/data/'
#             lines = scp_file.readlines()
#             for line in lines[:5]:
#
#                 tempt = str(line).split()[1]
#                 file_loc = tempt.split(':')[0][18:]  # mlp keep [18:]
#                 pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
#
#                 # Read the ark file to get utterance
#                 with open('../remote/data' + file_loc, 'rb') as ark_file:
#                     # use '../remote/data' + file_loc replace './data/' + file_loc
#                     ark_file.seek(int(pointer))
#                     utt_mat = kaldiark.parse_feat_matrix(ark_file)
#
#                     # Use model to get representations
#                     if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
#                         utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)
#
#                     # Init centers: randomly pick k data from data set as centers
#                     if start:
#                         centers = np.array(random.sample(list(utt_mat), k))  # k=4
#                         start = False
#                         print(centers.shape)
#
#                     # Assign data to clusters
#                     assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
#
#                     # Update centers
#                     for c_index in range(k):  # k=4
#                         data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
#                         # print(data_in_c.shape)
#                         if data_in_c.shape[0] > 0:  # some clusters may not have assigned data points
#                             centers[c_index] = np.mean(data_in_c, axis=0)
#
#                     # Calculate the clustering loss
#                     assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
#                     epoch_error += np.sum(assigns, axis=0)[1]
#
#         temp = epoch_error  # store the old assigns
#
#         end = time.time()
#         print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), epoch_error, (end-tmp)))
#         tmp = end
#         epochs += 1
#     total_loss.append(epoch_error)
#     print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, epoch_error, (end-checkpoint)))
# print("The best number of clusters is {:d}, the total trail costs {:0.7f} seconds".format((total_loss.index(min(total_loss))+1), (end-start_time)))
#
# # Draw the loss trend figure
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# y = total_loss
# x = np.arange(1, K+1)
# fig, ax = plt.subplots(figsize=(14,7))
# ax.plot(x,y,'r--',label='Cluster loss trend')
#
# ax.set_title('Clustering Loss with various cluster numbers',fontsize=18)
# ax.set_xlabel('Number of cluster', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
# ax.set_ylabel('Final clustering Loss', fontsize='x-large',fontstyle='oblique')
# ax.legend()
#
# plt.savefig("./kmeans/graph/{:s}_cluster_loss.pdf".format(NAME))










# class KMeans:
#
#     def __init__(self, k, data):
#         """
#         Run the K-means on given data with fixed number of clusters.
#         :param k: Fixed number of clusters predefined ahead.
#         :param data: Data for clustering.
#
#         :return centers: Centers of the clusters.
#         :return assigns: Correspondence between datapoint and cluster center with clustering errors.
#         """
#         self.k = k
#         self.data = data
#         self.centers = None
#         self.assigns = None
#         self.run_kmeans()
#
#     def eu_distance(self, a, b):
#         """
#         Calculate the Euclidean Distance between two point
#         :param a: Starting point
#         :param b: End Point
#         :return: Euclidean Distance
#         """
#         dist = np.sqrt(np.sum(np.square(a - b)))
#         # np.linalg.norm(a - b)  # numpy function can replace the above
#         return dist
#
#     def assign_cluster(self, datapoint):
#         """
#         Assign the given datapoint to a cluster center
#         :param datapoint: a data vector
#         :return: c_index: The assigned cluster index
#         :return: dist: The clustering error (L2 distance)
#         """
#         dists = np.array([self.eu_distance(c, datapoint) for c in self.centers])
#         c_index = np.argmin(dists)
#         dist = np.min(dists)
#         return c_index, dist
#
#     def run_kmeans(self):
#         """
#         Kernel steps of K-means
#         """
#         # Init centers
#         self.centers = np.array(random.sample(list(self.data), self.k))  # randomly pick k data from data set as centers
#
#         temp = None
#         while True:
#             # Assign data to clusters
#             assigns = np.array([self.assign_cluster(datapoint) for datapoint in self.data])
#
#             if (assigns == temp).all():  # satisfy the stop condition
#                 return
#
#             temp = assigns  # store the old assigns
#
#             # Updata centers
#             for c_index in range(self.k):
#                 data_in_c = np.array([self.data[i] for i in range(self.data.shape[0]) if assigns[i][0] == c_index])
#                 self.centers[c_index] = np.mean(data_in_c, axis=0)
#
#     def plot_result(self, num_classes=1):
#
#         plt.figure(figsize=(8, 6))
#         plt.title("Final centers in red")
#         plt.scatter(self.data[:, 0], self.data[:, 1], marker='.', c=num_classes)
#         plt.scatter(self.centers[:, 0], self.centers[:, 1], c='r', marker='x', s=500)
#         # plt.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1], c='k')
#         plt.show()
#
#
# if __name__ == '__main__':
#     from sklearn.datasets import make_blobs
#     np.random.seed(1000)
#     X, y = make_blobs(centers=4, n_samples=1000)  # generate samples
#
#     kmeans = KMeans(4, X)
#     kmeans.plot_result(y)
