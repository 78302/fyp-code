import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 

LEARNING_RATE = 0.01
EPOCH = 20

rnn = toy_lstm().to(device)  
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all parameters
loss_func = nn.MSELoss()
# Learning rate decay schedule
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)

# Train + Dev
train_loss = []
valid_loss = []
min_valid_loss = np.inf
for i in range(EPOCH):
    total_train_loss = []
    rnn.train()  # Training
    
    # Use the total scp files
    # Read data index from the total scp file
    with open('./data/raw_fbank_train_si284.scp', 'rb') as scp_file:
        lines = scp_file.readlines()
        for line in lines[:31]:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][18:]  # ark file path; keep [18:]
            pointer = temp.split(':')[1][:-3]  # pointer to the utterance

            # According to the file name and pointer to get the matrix
            with open('../remote/data' + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
            
                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)   # change data to tensor

                output = rnn(utt_mat)

                loss = loss_func(output, utt_mat)  # compute the difference
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # back-prop
                optimizer.step()  # gradients
                total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
    print('train complete!')

        

#     with open('./raw_fbank_train_si284.1.scp',
#               'rb') as scp_file:  # ../remote/data/wsj/fbank/raw_fbank_train_si284.10.scp
#         lines = scp_file.readlines()

#         count = 1  # batch counter
#         for line in lines[:64]:
#             tmp = str(line.strip()).strip("'").split()[1]
#             file_loc = tmp.split(':')[0][
#                        29:]  # path of the file containing the utterance, only need later part of the path # 18
#             pointer = tmp.split(':')[1]  # pointer of the utterance

#             # According to the file name and pointer to get the matrix
#             with open('./' + file_loc, 'rb') as ark_file:
#                 ark_file.seek(int(pointer))
#                 utter_mat = kaldiark.parse_feat_matrix(ark_file)

#                 utter_mat = np.expand_dims(utter_mat, axis=0)  # expand a new dimension as batch
#                 utter_mat = torch.Tensor(utter_mat).to(device)   # change data to tensor

#                 output = rnn(utter_mat)

#                 loss = loss_func(output, utter_mat)  # compute the difference
#                 optimizer.zero_grad()  # clear gradients for this training step
#                 loss.backward()  # back-propagation, compute gradients
#                 optimizer.step()  # apply gradients
#                 total_train_loss.append(loss.item())
#             train_loss.append(np.mean(total_train_loss))

    total_valid_loss = []
    rnn.eval()  # Validation
    
    # Use one of scp files
    # Read data index from the total scp file
    with open('./data/raw_fbank_train_si284.1.scp', 'rb') as scp_file:  # change 1 to dev 
        lines = scp_file.readlines()
        for line in lines[:7]:
            temp = str(line).split()[1]
            file_loc = temp.split(':')[0][18:]  # ark file path; keep [18:]
            pointer = temp.split(':')[1][:-3]  # pointer to the utterance

            # According to the file name and pointer to get the matrix
            with open('../remote/data' + file_loc, 'rb') as ark_file:
                ark_file.seek(int(pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
            
                utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
                utt_mat = torch.Tensor(utt_mat).to(device)   # change data to tensor

                with torch.no_grad():
                    output = rnn(utt_mat)  # rnn output

                loss = loss_func(output, utt_mat)
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
    print('dev complete!')
    
    
    
    
#     with open('./raw_fbank_train_si284.2.scp',
#               'rb') as scp_file:  # ../remote/data/wsj/fbank/raw_fbank_train_si284.10.scp
#         lines = scp_file.readlines()
#         for line in lines[:16]:
#             tmp = str(line.strip()).strip("'").split()[1]
#             file_loc = tmp.split(':')[0][29:]  # path of the file containing the utterance,
#             # only need later part of the path # 18
#             pointer = tmp.split(':')[1]  # pointer of the utterance

#             # According to the file name and pointer to get the matrix
#             with open('./' + file_loc, 'rb') as ark_file:
#                 ark_file.seek(int(pointer))
#                 utter_mat = kaldiark.parse_feat_matrix(ark_file)

#                 utter_mat = np.expand_dims(utter_mat, axis=0)  # expand a new dimension as batch
#                 utter_mat = torch.Tensor(utter_mat).to(device)   # change data to tensor

#                 with torch.no_grad():
#                     output = rnn(utter_mat)  # rnn output

#                 loss = loss_func(output, utter_mat)
#             total_valid_loss.append(loss.item())
#         valid_loss.append(np.mean(total_valid_loss))
#     # print("Val-complete")

    if (valid_loss[-1] < min_valid_loss):
        torch.save({'epoch': i, 'model': rnn, 'train_loss': train_loss,
                    'valid_loss': valid_loss}, './LSTM.model')
        min_valid_loss = valid_loss[-1]

    # Log
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                  'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  valid_loss[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])
    mult_step_scheduler.step()  # 学习率更新
    print(log_string)  # 打印日志
