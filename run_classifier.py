import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
from classifier import classification_net
import glob

import argparse
# Deal with the use input parameters
# ideal input: epoch, learning-rate, K, input-size, hidden-size, train-scp-path, dev-scp-path, layers
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--learning_rate', '-lr', help='Learning rate, not required', type=float, default=0.1)
parser.add_argument('--epoch', '-e', help='Epoch, not required', type=int, default=100)
parser.add_argument('--gap', '-k', help='Position in origin where the first prediction correspond to, not required', type=int,  default=100)
parser.add_argument('--input_size', '-is', help='Input dimension, not required', type=float, default=40)
parser.add_argument('--hidden_size', '-hs', help='Hidden vector dimension, not required', type=float, default=512)
parser.add_argument('--output_size', '-os', help='Output dimension, not required', type=float, default=43)
parser.add_argument('--model_path', '-p', help='Path of the pre-trained model, not required', default=None)
parser.add_argument('--train_utt', '-tu', help='Path of the input training utterance scp file, not required', default='./data/si284-0.9-train.fbank.scp')
parser.add_argument('--train_label', '-tl', help='Path of the input training label scp file, not required', default='./data/si284-0.9-train.bpali.scp')
parser.add_argument('--train_phone', '-tp', help='Path of the input training label scp file, not required', default='./data/train-si284.bpali')
parser.add_argument('--dev_utt', '-du', help='Path of the input validation utterance scp file, not required', default='./data/si284-0.9-train.fbank.scp')
parser.add_argument('--dev_label', '-dl', help='Path of the input validation label scp file, not required', default='./data/si284-0.9-train.bpali.scp')
parser.add_argument('--dev_phone', '-dp', help='Path of the input training label scp file, not required', default='./data/train-si284.bpali')
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
LEARNING_RATE = args.learning_rate
EPOCH = args.epoch
K = args.gap
INPUT_SIZE = args.input_size
HIDDEN_SIZE = args.hidden_size
OUTPUT_SIZE = args.output_size
PRETRAIN_PATH = args.model_path
# TRAIN_UTT_SCP_PATH = eval(args.train_utt)
# TRAIN_LABEL_SCP_PATH = eval(args.train_label)
# TRAIN_LABEL_PATH = eval(args.train_phone)
# DEV_UTT_SCP_PATH = eval(args.dev_utt)
# DEV_LABEL_SCP_PATH = eval(args.dev_label)
# DEV_LABEL_PATH = eval(args.dev_phone)

TRAIN_UTT_SCP_PATH = args.train_utt
TRAIN_LABEL_SCP_PATH = args.train_label
TRAIN_LABEL_PATH = args.train_phone
DEV_UTT_SCP_PATH = args.dev_utt
DEV_LABEL_SCP_PATH = args.dev_label
DEV_LABEL_PATH = args.dev_phone

with open('./data/phones-unk.txt', 'r') as ph_file:
    standard = ph_file.read().splitlines()
    # print(standard)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #
# classifier use representation dimension as input size -- i.e. HIDDEN_SIZE
classifier = classification_net(INPUT_SIZE=INPUT_SIZE, HIDDEN_SIZE=HIDDEN_SIZE, OUTPUT_SIZE=OUTPUT_SIZE, PRETRAIN_PATH=PRETRAIN_PATH).to(DEVICE)
# print(net)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LEARNING_RATE)  # optimize all require grad's parameters
# Learning rate decay schedule
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)
loss_func = torch.nn.CrossEntropyLoss()

train_bpali_file = open(TRAIN_LABEL_PATH, 'rb')
train_fbank_scp = open(TRAIN_UTT_SCP_PATH, 'rb')
dev_bpali_file = open(DEV_LABEL_PATH, 'rb')
dev_fbank_scp = open(DEV_UTT_SCP_PATH, 'rb')
train_fbank_lines = train_fbank_scp.readlines()
dev_fbank_lines = dev_fbank_scp.readlines()

# Train + Dev
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
min_valid_loss = np.inf
for i in range(EPOCH):
    total_train_loss = []
    classifier.train()  # Training

    train_correct = 0
    train_total = 0
    with open(TRAIN_LABEL_SCP_PATH, 'rb') as scp_file:
        bpali_lines = scp_file.readlines()
        for idx,line in enumerate(bpali_lines[:K]):
            # Find the label
            temp = str(line).split()[1]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the label
            # Linux only has \n
            train_bpali_file.seek(int(pointer))
            transcript = train_bpali_file.readline()
            labels = str(transcript)[2:-3].split()
            # print(len(labels))

            # Find the utterance
            utt_line = train_fbank_lines[idx]
            temp = str(utt_line).split()[1]
            utt_file_loc = temp.split(':')[0][14:]  # ubuntu [24:] mlp [14:]
            utt_pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(temp, utt_file_loc)

            with open('../remote/data' + utt_file_loc, 'rb') as ark_file:
                # mlp use: '../remote/data' + utt_file_loc
                # ubuntu use: './data' + utt_file_loc
                ark_file.seek(int(utt_pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
                # print(utt_mat.shape )
                utt_mat = torch.Tensor(utt_mat).to(DEVICE)   # change data to tensor

                for iidx in range(utt_mat.shape[0]):
                    mat = utt_mat[iidx]
                    mat = torch.unsqueeze(mat, 0)
                    x = mat.to(DEVICE)
                    y = standard.index(labels[iidx])
                    y = torch.tensor([y], dtype=torch.long).to(DEVICE)

                    optimizer.zero_grad()
                    output=classifier(x)

                    # print(torch.equal(output[0].argmax(), y[0]))

                    if torch.equal(output[0].argmax(), y[0]):
                        train_correct += 1
                    train_total += 1

                    loss=loss_func(output,y)

                    loss.backward()
                    optimizer.step()
                    total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))
    train_acc.append(train_correct/train_total)

    total_valid_loss = []
    classifier.eval()  # Validation

    dev_correct = 0
    dev_total = 0
    with open(DEV_LABEL_SCP_PATH, 'rb') as scp_file:
        bpali_lines = scp_file.readlines()
        for didx,line in enumerate(bpali_lines[:K//2]):
            # Find the label
            temp = str(line).split()[1]
            pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the label
            # Linux only has \n
            dev_bpali_file.seek(int(pointer))
            transcript = dev_bpali_file.readline()
            labels = str(transcript)[2:-3].split()
            # print(pointer, len(labels))

            # Find the utterance
            utt_line = dev_fbank_lines[didx]
            temp = str(utt_line).split()[1]
            utt_file_loc = temp.split(':')[0][14:]  # ark file path; keep [14:]
            utt_pointer = temp.split(':')[1][:-3].replace('\\r', '')  # pointer to the utterance
            # print(didxutt_file_loc, utt_pointer)
            # According to the file name and pointer to get the matrix
            with open('../remote/data' + utt_file_loc, 'rb') as ark_file:
                # mlp use: '../remote/data' + utt_file_loc
                # ubuntu use: './data' + utt_file_loc
                ark_file.seek(int(utt_pointer))
                utt_mat = kaldiark.parse_feat_matrix(ark_file)
                # print(utt_mat.shape)
                utt_mat = torch.Tensor(utt_mat).to(DEVICE)   # change data to tensor
#                 utt_mat = np.expand_dims(utt_mat, axis=0)  # expand a new dimension as batch
#             print(utt_mat.shape)

                for diidx in range(utt_mat.shape[0]):
                    mat = utt_mat[diidx]
                    mat = torch.unsqueeze(mat, 0)
                    x = mat.to(DEVICE)
                    # print(len(labels), iidx)
                    y = standard.index(labels[diidx])
                    # print(utt_pointer, len(labels), iidx)
                    y = torch.tensor([y], dtype=torch.long).to(DEVICE)

                    with torch.no_grad():
                        output = classifier(x)

                        if torch.equal(output[0].argmax(), y[0]):
                            dev_correct += 1
                        dev_total += 1

                    loss=loss_func(output,y)
                    total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))
    valid_acc.append(dev_correct/dev_total)

    # save the net

    min_valid_loss = np.min(valid_loss)

    if ((i + 1) % 1 == 0):
        torch.save({'epoch': i + 1, 'state_dict': classifier.state_dict(), 'train_loss': train_loss,
                    'valid_loss': valid_loss, 'optimizer': optimizer.state_dict()},
                    './model_classifier/Epoch{:d}_{:s}.pth.tar'.format((i + 1), NAME))


    # Log
    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, train_acc: {:0.6f}, valid_loss: {:0.6f}, '
                  'valid_acc: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), EPOCH,
                                                                  train_loss[-1],
                                                                  train_acc[-1],
                                                                  valid_loss[-1],
                                                                  valid_acc[-1],
                                                                  min_valid_loss,
                                                                  optimizer.param_groups[0]['lr'])
    mult_step_scheduler.step()  # 学习率更新
    print(log_string)  # 打印日志

train_bpali_file.close()
train_fbank_scp.close()
dev_bpali_file.close()
dev_fbank_scp.close()
