# Implementation of K-means
import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob
import matplotlib.pyplot as plt
from functions import assign_cluster, pretrain_representations, closest_centroid
import random

# Load kmeans paras
import argparse
# Deal with the use input parameters
# ideal input: name, epoch, K, pretrain_model_path, hidden-size, train-scp-path, dev-scp-path, layers
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=1)
parser.add_argument('--type', '-t', help='Ubuntu type or mlp type, not required default is Ubuntu type', type=int,  default=1)
parser.add_argument('--cluster_number', '-k', help='The largest number of clusters, not required', type=int,  default=10)
parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)  # model path: './pretrain_model/model/Epoch50.pth.tar'
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
EPOCH = args.epoch
TYPE = args.type
K = args.cluster_number
PRETRAIN_PATH = args.model_path


# Decide the file path under different environment
# Python do not have switch case, use if else instead
if TYPE == 1:  # under Ubbuntu test environment
    SCP_FILE = './data/si284-0.9-train.fbank.scp'  # scp file path under Ubuntu environment
    UTT_RELATIVE_PATH = './data/'  # relative path of ark file under Ubuntu environment
    C = 24  # cutting position to divide the list
else:
    SCP_FILE = '../remote/data/wsj/extra/si284-0.9-train.fbank.scp'
    UTT_RELATIVE_PATH = '../remote/data'
    C = 14


np.random.seed(100)
start = False

import time
start_time = time.time()
tmp = start_time

epochs = 0
temp = None
start = True
k=K


# Load ceters and loss
import csv

for e in range(EPOCH):

    error = 0
    # assign_time = 0
    # record_time = 0
    d_centers = np.zeros(k)
    if PRETRAIN_PATH:
        n_centers = np.zeros((k, 512))
    else:
        n_centers = np.zeros((k, 40))

    with open(SCP_FILE, 'rb') as scp_file:
        lines = scp_file.readlines()
        # for utterance in the file
        for line in lines:  # use 2 for test
            tempt = str(line).split()[1]
            file_loc = tempt.split(':')[0][C:]
            pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

            with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

            # Use pretrain model to get representations
            if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
                utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)

            # Init centers:
            # randomly pick k data from data set as centers
            if start:
                # if k <= utt_mat.shape[0]:
                #     centers = np.array(random.sample(list(utt_mat), k))  # k=4
                # else:
                u_max = np.max(utt_mat,axis=0)
                u_min = np.min(utt_mat,axis=0)
                # print(u_max.shape)

                centers = np.random.rand(k, utt_mat.shape[1])
                centers = (u_max - u_min) * centers + u_min

                start = False
                print(centers.shape)

            # Assign centers to the utterance
            assigns, errors = closest_centroid(utt_mat, centers)
            # end = time.time()
            # assign_time += end-tmp
            # tmp = end
            error += np.sum(errors)
            # print(error)

            # Record number of frames and f information
            for i in range(utt_mat.shape[0]):
                c = int(assigns[i])
                # n_centers[c] = d_centers[c]/(d_centers[c]+1) * n_centers[c] + 1/(d_centers[c]+1) * utt_mat[i]
                n_centers[c] += utt_mat[i]
                d_centers[c] += 1
            # end = time.time()
            # record_time += end-tmp
            # tmp = end

        # print(n_centers.shape)
        # print(d_centers)

        # Update Centers
        for c in range(k):
            if d_centers[c] > 0:
                centers[c] = n_centers[c] / d_centers[c]

        end = time.time()
        error = error / np.sum(d_centers)
        print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), error, (end-tmp)))
        tmp = end

        # write to the csv file for each epoch
        with open(NAME + '_result.csv', 'a+') as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow([k, epochs+1, error])
        epochs += 1
print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, error, (end-start_time)))






# # Implementation of K-means
# import torch
# import torch.nn as nn
# import numpy as np
#
# import kaldiark
# from apc import toy_lstm
# import glob
# import matplotlib.pyplot as plt
# from functions import assign_cluster, pretrain_representations
# import random
#
# # Load kmeans paras
# import argparse
# # Deal with the use input parameters
# # ideal input: name, epoch, K, pretrain_model_path, hidden-size, train-scp-path, dev-scp-path, layers
# parser = argparse.ArgumentParser(description='Parse the net paras')
# parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
# parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=1)
# parser.add_argument('--type', '-t', help='Ubuntu type or mlp type, not required default is Ubuntu type', type=int,  default=1)
# parser.add_argument('--cluster_number', '-k', help='The largest number of clusters, not required', type=int,  default=43)
# parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)  # model path: './pretrain_model/model/Epoch50.pth.tar'
# # required=True
# args = parser.parse_args()
#
# # Assign parameters
# NAME = args.name
# EPOCH = args.epoch
# TYPE = args.type
# K = args.cluster_number
# PRETRAIN_PATH = args.model_path
#
# # Decide the file path under different environment
# # Python do not have switch case, use if else instead
# if TYPE == 1:  # under Ubbuntu test environment
#     SCP_FILE = './data/si284-0.9-train.fbank.scp'  # scp file path under Ubuntu environment
#     UTT_RELATIVE_PATH = './data/'  # relative path of ark file under Ubuntu environment
#     C = 24  # cutting position to divide the list
# else:
#     SCP_FILE = '../remote/data/wsj/extra/si284-0.9-train.fbank.scp'
#     UTT_RELATIVE_PATH = '../remote/data'
#     C = 14
#
# # print(NAME, EPOCH, TYPE, K, SCP_FILE, UTT_RELATIVE_PATH, C)
# # exit()
#
# random.seed(100)
# start = False
#
# import time
# start_time = time.time()
# tmp = start_time
#
# epochs = 0
# temp = None
# k=K
#
#
# # Load ceters and loss
# import csv
# try:
#     centers = np.load(NAME + '_centers.npy')
#     if centers.shape[0] != k:
#         start = True
#     with open(NAME + '_result.csv', 'r') as csv_file:
#         interm_data = []
#         csvreader = csv.reader(csv_file)
#         for l in csvreader:
#             interm_data.append(l)
#         if len(interm_data) > 0 and int(interm_data[-1][0]) == k:  # not empty, load the loss
#             temp = float(interm_data[-1][2])
#         else:  # Need to initialize centers
#             start = True
#
# except: # Need to initialize centers
#     start = True
#
# print(start)
#
# # Start Kmeans procedure
#
#
#
# for e in range(EPOCH):
#     epoch_error = 0
#
#     d_centers = np.zeros(k)
#     if PRETRAIN_PATH:
#         n_centers = np.zeros((k, 512))
#     else:
#         n_centers = np.zeros((k, 40))
#
#
#     # Read the SCP file
#     with open(SCP_FILE, 'rb') as scp_file:
#         lines = scp_file.readlines()
#         for line in lines[:1]:  # remove [:K]
#             tempt = str(line).split()[1]
#             file_loc = tempt.split(':')[0][C:]
#             pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
#
#             # Read the ark file to get utterance
#             with open(UTT_RELATIVE_PATH + file_loc, 'rb') as ark_file:
#                 ark_file.seek(int(pointer))
#                 utt_mat = kaldiark.parse_feat_matrix(ark_file)
#
#                 # Use model to get representations
#                 if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
#                     utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)
#
#                 # Init centers: randomly pick k data from data set as centers
#                 if start:
#
#                     if k <= utt_mat.shape[0]:
#                         centers = np.array(random.sample(list(utt_mat), k))  # k=4
#                     else:
#                         u_max = np.max(utt_mat,axis=0)
#                         u_min = np.min(utt_mat,axis=0)
#
#                         centers = np.random.rand(k, utt_mat.shape[1])
#                         centers = (u_max - u_min) * centers + u_min
#
#                     start = False
#                     # print(centers)
#                     print(centers.shape, d_centers.shape, n_centers.shape)
#
#                 # Assign data to clusters
#                 assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
#
#                 # Record data information
#                 # count = 0
#                 for i in range(utt_mat.shape[0]):
#                     c = int(assigns[i][0])
#                     # print(c)
#                     # print(utt_mat[i][:2])
#                     # print(n_centers[c][:2], d_centers[c])
#                     # n_centers[c] = d_centers[c]/(d_centers[c]+1) * n_centers[c] + 1/(d_centers[c]+1) * utt_mat[i]
#
#                     n_centers[c] += utt_mat[i]
#                     d_centers[c] += 1
#
#                     # count += 1
#                     # if count == 6:
#                     #     break
#
#                 # print(n_centers, d_centers)
#
#
#
#
#                 # for c in range(k):
#                 #     for i in range():
#                 #         if assigns[i][0] == c:
#                 #             n_centers[c] = d_centers[c]/(d_centers[c]+1) * n_centers[c] + 1/(d_centers[c]+1) * utt_mat[i]
#                 #             d_centers[c] += 1
#
#
#                 # # Update centers
#                 # for c_index in range(k):  # k=4
#                 #     data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
#                 #     # print(data_in_c.shape)
#                 #     if data_in_c.shape[0] > 0:  # some clusters may not have assigned data points
#                 #         centers[c_index] = np.mean(data_in_c, axis=0)
#
#         # Calculate the clustering loss
#         epoch_error += np.sum(assigns, axis=0)[1]
#
#         # After iterate the whole file
#         # update the centers
#         for c in range(k):
#             if d_centers[c] > 0:
#                 print(centers[c][:2], n_centers[c][:2], d_centers[c])
#                 centers[c] = n_centers[c] / d_centers[c]
#                 print(centers[c][:2])
#         # print(d_centers, n_centers)
#
#     end = time.time()
#     print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), epoch_error / np.sum(d_centers), (end-tmp)))
#     epoch_error = epoch_error / np.sum(d_centers)
#     tmp = end
#     epochs += 1
#
#     # write to the csv file for each epoch
#     np.save(NAME + '_centers.npy', centers)
#     with open(NAME + '_result.csv', 'a+') as csv_file:
#         csvwriter = csv.writer(csv_file)
#         csvwriter.writerow([k, epochs, epoch_error])
#
#     # Judge whether it can be free from the loop
#     if temp:
#         if temp == epoch_error:
#             break
#     temp = epoch_error
#
# print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, epoch_error, (end-start_time)))
#
#


'''
# if L:
#     for k in range(1, K+1):
#         start = True
#         epochs = 0
#         temp = None
#         checkpoint = tmp
#         for e in range(EPOCH):
#             epoch_error = 0
#
#             # Judge whether it can be free from the loop
#             # use if condition
#             # --------------
#             # #################
#
#             # Read the SCP file
#             with open(SCP_FILE, 'rb') as scp_file:
#                 # mlp use '../remote/data/wsj/fbank/' replace '/data/'
#                 lines = scp_file.readlines()
#                 for line in lines:
#
#                     tempt = str(line).split()[1]
#                     file_loc = tempt.split(':')[0][18:]  # mlp keep [18:]
#                     pointer = tempt.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
#
#                     # Read the ark file to get utterance
#                     with open('../remote/data' + file_loc, 'rb') as ark_file:
#                         # use '../remote/data' + file_loc replace './data/' + file_loc
#                         ark_file.seek(int(pointer))
#                         utt_mat = kaldiark.parse_feat_matrix(ark_file)
#
#                         # Use model to get representations
#                         if PRETRAIN_PATH:  # './pretrain_model/model/Epoch50.pth.tar'
#                             utt_mat = pretrain_representations(PRETRAIN_PATH, utt_mat)
#
#                         # Init centers: randomly pick k data from data set as centers
#                         if start:
#                             centers = np.array(random.sample(list(utt_mat), k))  # k=4
#                             start = False
#                             print(centers.shape)
#
#                         # Assign data to clusters
#                         assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
#
#                         # Update centers
#                         for c_index in range(k):  # k=4
#                             data_in_c = np.array([utt_mat[i] for i in range(utt_mat.shape[0]) if assigns[i][0] == c_index])
#                             # print(data_in_c.shape)
#                             if data_in_c.shape[0] > 0:  # some clusters may not have assigned data points
#                                 centers[c_index] = np.mean(data_in_c, axis=0)
#
#                         # Calculate the clustering loss
#                         assigns = np.array([assign_cluster(datapoint, centers) for datapoint in utt_mat])
#                         epoch_error += np.sum(assigns, axis=0)[1]
#
#             temp = epoch_error  # store the old assigns
#
#             end = time.time()
#             print("Epoch {:d} error: {:0.7f}, use {:0.7f} seconds.".format((epochs+1), epoch_error, (end-tmp)))
#             tmp = end
#             epochs += 1
#         total_loss.append(epoch_error)
#         print("{:d} clusters, final loss is {:0.7f}, costs {:0.7f} seconds".format(k, epoch_error, (end-checkpoint)))
#     print("The best number of clusters is {:d}, the total trail costs {:0.7f} seconds".format((total_loss.index(min(total_loss))+1), (end-start_time)))
#
#     # Draw the loss trend figure
#     import numpy as np
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     y = total_loss
#     x = np.arange(1, K+1)
#     fig, ax = plt.subplots(figsize=(14,7))
#     ax.plot(x,y,'r--',label='Cluster loss trend')
#
#     ax.set_title('Clustering Loss with various cluster numbers',fontsize=18)
#     ax.set_xlabel('Number of cluster', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
#     ax.set_ylabel('Final clustering Loss', fontsize='x-large',fontstyle='oblique')
#     ax.legend()
#
#     plt.savefig("./kmeans/graph/{:s}_cluster_loss.pdf".format(NAME))
#
# else:
#     start = True
#     epochs = 0
#     temp = None
#     checkpoint = tmp
#     k=K
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
#             for line in lines:
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
'''

'''
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

'''
