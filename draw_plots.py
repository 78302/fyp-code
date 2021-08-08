# # Load data file paras
# import argparse
# # Deal with the use input parameters
# parser = argparse.ArgumentParser(description='Parse the net paras')
# parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
# parser.add_argument('--target', '-t', help='Name of the Model, required', type=int, required=True)
# # required=True
# args = parser.parse_args()
#
# # Assign parameters
# NAME = args.name
# TARGET = args.target
#
# if TARGET == 1:
#     #### Kmeans plot
#     # import data from csv_file
#     import csv
#     # Load ceters and loss
#     import csv
#     cluster_bonds = []
#     loss = []
#     with open(NAME + '_result.csv', 'r') as csv_file:
#         csvreader = csv.reader(csv_file)
#         k = 0
#         error = 0
#         for l in csvreader:
#             if int(l[0]) != k:
#                 cluster_bonds.append(k)
#                 loss.append(error)
#             k = int(l[0])
#             error = float(l[2])
#         cluster_bonds.append(k)
#         loss.append(error)
#
#
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     x = cluster_bonds[1:]
#     y = loss[1:]
#
#     fig, ax = plt.subplots(figsize=(14,7))
#     ax.plot(x,y,'r--',label='Loss')
#
#     ax.set_title('Loss',fontsize=18)
#     ax.set_xlabel('Number of Clusters', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
#     ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
#     ax.legend()
#
#     plt.savefig("Clustering Loss.pdf")
#
# # print(cluster_bonds, loss)
#
#
#
#
# if TARGET == 2:
#     ### DRAW classifier loss curve
#     # Draw the train loss
#     import numpy as np
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#     # load data
#     e_results = np.load(NAME + '_results.npy')
#
#     y1 = e_results[0]
#     y2 = e_results[1]
#     x = np.arange(1,21)
#     fig, ax = plt.subplots(figsize=(14,7))
#     ax.plot(x,y1,'r',label='Train Loss')
#     ax.plot(x,y2,'b',label='Dev Loss')
#
#     ax.set_title('Loss',fontsize=18)
#     ax.set_xlabel('Epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
#     ax.set_ylabel('Loss', fontsize='x-large',fontstyle='oblique')
#     ax.legend()
#
#     plt.savefig("{:s}_classifier_loss.pdf".format(NAME))
#
#     y1 = e_results[2]
#     y2 = e_results[3]
#     x = np.arange(1,21)
#     fig2, ax2 = plt.subplots(figsize=(14,7))
#     ax2.plot(x,y1,'r--',label='Train Err')
#     ax2.plot(x,y2,'r--',label='Dev Err')
#
#     ax2.set_title('Error Rate',fontsize=18)
#     ax2.set_xlabel('Epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
#     ax2.set_ylabel('Error Rate', fontsize='x-large',fontstyle='oblique')
#     ax2.legend()
#
#     plt.savefig("{:s}_classifier_err.pdf".format(NAME))
#
#
#
# if TARGET == 3:
#     ### DRAW apc loss curve
#     # Draw the train loss
#     import numpy as np
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#     # Load _losses
#     losses = np.load(NAME + '_losses.npy')
#
#     y_t = losses[0]
#     y_d = losses[1]
#     x = np.arange(1,101)
#     fig, ax = plt.subplots(figsize=(14,7))
#     ax.plot(x,y_t,'r',label='Train Loss')
#     ax.plot(x,y_d,'b',label='Dev Loss')
#
#     ax.set_title('Loss',fontsize=18)
#     ax.set_xlabel('epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
#     ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
#     ax.legend()
#
#     plt.savefig("{:s}_apc_loss.pdf".format(NAME))




import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
from vqapc import toy_vqapc
import matplotlib.pyplot as plt
import glob


# Model loading
def load_model(path, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


PATH_apc = './pretrain_model/model/Epoch50.pth.tar'  # Epoch100_WSJ_APC_lr0001.pth.tar
PATH_vqapc = './pretrain_model/model/Epoch40_WSJ_VQAPC_50epochs.pth.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 对照-使用默认参数
rnn_raw = toy_lstm(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4).to(device)
optimizer_raw = torch.optim.Adam(rnn_raw.parameters(), lr=0.001)  # optimize all parameters

# 使用预训练参数
rnn_pretrain = toy_lstm(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4).to(device)
optimizer_pretrain = torch.optim.Adam(rnn_pretrain.parameters(), lr=0.001)  # optimize all parameters
rnn_pretrain, optimizer_pretrain = load_model(PATH_apc, rnn_pretrain, optimizer_pretrain)

# 使用VQAPC
rnn_preVQ = toy_vqapc(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4).to(device)
optimizer_preVQ = torch.optim.Adam(rnn_preVQ.parameters(), lr=0.001)  # optimize all parameters
rnn_preVQ, optimizer_preVQ = load_model(PATH_vqapc, rnn_preVQ, optimizer_preVQ)

rnn_preVQ.eval()
rnn_pretrain.eval()
rnn_raw.eval()

K = 2  # predefine the gap

ori_mat = []
pre_mat = []
with open('./data/raw_fbank_train_si284.1.scp', 'rb') as scp_file:
    lines = scp_file.readlines()
    # for line in lines[:2]:  # use 1 utt to test
    temp = str(lines[0]).split()[1]
    file_loc = temp.split(':')[0][28:]  # ark file path; keep [18:]
    pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

    # According to the file name and pointer to get the matrix
    with open('./data' + file_loc, 'rb') as ark_file:
        ark_file.seek(int(pointer))
        utt_mat = kaldiark.parse_feat_matrix(ark_file)

        utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
        utt_mat = torch.Tensor(utt_mat).to(device)   # change data to tensor

        output_raw = rnn_raw(utt_mat[:, :-K, :])
        output_pretrain = rnn_pretrain(utt_mat[:, :-K, :])
        output_preVQ = rnn_preVQ(utt_mat[:, :-K, :], True)

        ori_mat.append(utt_mat[0, :-K, :])
        pre_mat.append(output_raw[0])
        pre_mat.append(output_pretrain[0])
        pre_mat.append(output_preVQ[0])

m1 = ori_mat[0].numpy()
m2 = pre_mat[0].detach().numpy()
m3 = pre_mat[1].detach().numpy()
m4 = pre_mat[2].detach().numpy()

# Save Image Function
fig = plt.figure(figsize=(10,8))
ax = plt.gca()
cax = plt.imshow(m1[50:100], cmap='viridis')
# plt.title('Original Utterance',fontsize=18, fontweight = 'bold')
plt.xticks(fontsize=20, fontweight = 'bold')
plt.yticks(fontsize=20, fontweight = 'bold')
plt.savefig('origin.pdf', dpi = 300)

# Save Image Function
fig = plt.figure(figsize=(10,8))
ax = plt.gca()
cax = plt.imshow(m3[50:100], cmap='viridis')
# plt.title('Recovered Utterance from APC',fontsize=18, fontweight = 'bold')
plt.xticks(fontsize=20, fontweight = 'bold')
plt.yticks(fontsize=20, fontweight = 'bold')
plt.savefig('pretrained.pdf', dpi = 300)

# Save Image Function
fig = plt.figure(figsize=(10,8))
ax = plt.gca()
cax = plt.imshow(m4[50:100], cmap='viridis')
plt.xticks(fontsize=12, fontweight = 'bold')
plt.yticks(fontsize=12, fontweight = 'bold')
plt.savefig('preVQ.pdf', dpi = 300)
