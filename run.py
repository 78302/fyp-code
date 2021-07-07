# print('Hello world!')

# '''
import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 

LEARNING_RATE = 0.001
EPOCH = 50  # change to 100 when using sbatch

rnn = toy_lstm().to(device)  
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters
loss_func = nn.MSELoss()
# Learning rate decay schedule
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.5)

K = 2
# Train + Dev
train_loss = []
valid_loss = []
min_valid_loss = np.inf

import time
start = time.time()
tmp = start

for i in range(EPOCH):
    total_train_loss = []
    rnn.train()  # Training

    # Use the total scp files
    # Read data index from the total scp file
    with open('./data/raw_fbank_train_si284.scp', 'rb') as scp_file:
        # mlp file path: ./data/raw_fbank_train_si284.scp
        # test file path: ./data/raw_fbank_train_si284.1.scp
        lines = scp_file.readlines()
        for line in lines:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][18:]  # ark file path; keep [18:]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(file_loc, pointer)

            # According to the file name and pointer to get the matrix
            with open('../remote/data' + file_loc, 'rb') as ark_file:
                # mlp file path: '../remote/data' + file_loc
                # test file path: '../fyp_code/data' + file_loc
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

                output = rnn(utt_mat[:, :-K, :])

                #                 print(utt_mat.shape, output.shape)

                loss = loss_func(output, utt_mat[:, K:, :])  # compute the difference
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # back-prop
                optimizer.step()  # gradients
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
    # print('train complete!')

    total_valid_loss = []
    rnn.eval()  # Validation

    # Use one of scp files
    # Read data index from the total scp file
    with open('./data/raw_fbank_dev.scp', 'rb') as scp_file:  
        # mlp file path: ./data/raw_fbank_dev.scp
        # test file path: ./data/raw_fbank_train_si284.2.scp
        lines = scp_file.readlines()
        for line in lines:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][18:]  # ark file path; keep [18:]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

            # According to the file name and pointer to get the matrix
            with open('../remote/data' + file_loc, 'rb') as ark_file:
                # mlp file path: '../remote/data' + file_loc
                # test file path: '../fyp_code/data' + file_loc
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)

                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)  # change data to tensor

                with torch.no_grad():
                    output = rnn(utt_mat[:, :-K, :])  # rnn output

                #                     print(utt_mat_mat.shape, output.shape)

                loss = loss_func(output, utt_mat[:, K:, :])
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
    # print('dev complete!')

#     if (valid_loss[-1] < min_valid_loss):
#         torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
#                     'valid_loss': valid_loss}, './LSTM.model')
#         min_valid_loss = valid_loss[-1]
    
    min_valid_loss = np.min(valid_loss)
    
    end = time.time()
    
    # save the net
    if ((i + 1) % 10 == 0):
        torch.save({'epoch': i + 1, 'state_dict': rnn.state_dict(), 'train_loss': train_loss,
                    'valid_loss': valid_loss, 'optimizer': optimizer.state_dict()},
                    './model/Epoch{:d}.pth.tar'.format((i + 1), min_valid_loss))

    # Log
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}, time: {:0.7f}').format((i + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'],
                                                                  (end-tmp))
    tmp = end
    mult_step_scheduler.step()  # 学习率更新
    print(log_string)  # 打印日志
    
print(end-start)
    
    
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
y = train_loss
x = np.arange(0,len(train_loss))
fig, ax = plt.subplots(figsize=(14,7))
ax.plot(x,y,'r--',label='type1')

ax.set_title('Loss',fontsize=18)
ax.set_xlabel('epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
ax.legend()

plt.savefig("loss.pdf")


# def load_model(path, model, optimizer):
    
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return model, optimizer

# PATH = './model/Epoch20.pth.tar'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 对照-使用默认参数
# rnn_raw = toy_lstm().to(device)
# optimizer_raw = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters

# # 使用预训练参数
# rnn_pretrain = toy_lstm().to(device)
# optimizer_pretrain = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters
# rnn_pretrain, optimizer_pretrain = load_model(PATH, rnn_pretrain, optimizer_pretrain)

# rnn_pretrain.eval()
# rnn_raw.eval()

# # get 2 utt mats:
# # Predefine the prediction gap
# K = 2  # predefine the gap

# ori_mat = []
# pre_mat = []
# with open('./data/raw_fbank_train_si284.1.scp', 'rb') as scp_file:
#     lines = scp_file.readlines()
#     # for line in lines[:2]:  # use 1 utt to test
#     temp = str(lines[0]).split()[1]
#     file_loc = temp.split(':')[0][28:]  # ark file path; keep [18:]
#     pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance

#     # According to the file name and pointer to get the matrix
#     with open('./data' + file_loc, 'rb') as ark_file:
#         ark_file.seek(int(pointer))
#         utt_mat = kaldiark.parse_feat_matrix(ark_file)
            
#         utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
#         utt_mat = torch.Tensor(utt_mat).to(device)   # change data to tensor
        
#         output_raw = rnn_raw(utt_mat[:, :-K, :])
#         output_pretrain = rnn_pretrain(utt_mat[:, :-K, :])
                
#         ori_mat.append(utt_mat[0, :-K, :])
#         pre_mat.append(output_raw[0])
#         pre_mat.append(output_pretrain[0])
        
# m1 = ori_mat[0].cpu().numpy()
# m2 = pre_mat[0].cpu().detach().numpy()
# m3 = pre_mat[1].cpu().detach().numpy()

# # Save Image Function
# fig = plt.figure(figsize=(10,8))
# ax = plt.gca()
# cax = plt.imshow(m1, cmap='viridis')
# plt.savefig('origin.pdf')

# cax = plt.imshow(m2, cmap='viridis')
# plt.savefig('raw.pdf')

# cax = plt.imshow(m3, cmap='viridis')
# plt.savefig('pretrained.pdf')

