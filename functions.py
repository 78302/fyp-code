import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob

# Model loading
def load_model(path, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

if __name__ == '__main__':

    from apc import toy_lstm

    rnn_pretrain_model = toy_lstm(INPUT_SIZE=40, HIDDEN_SIZE=512, LAYERS=4)
    optimizer_pretrain = torch.optim.Adam(rnn_pretrain_model.parameters(), lr=0.001)  # just attach but no use
    rnn_pretrain_model, optimizer_pretrain = load_model('./model/Epoch1.pth.tar', rnn_pretrain_model, optimizer_pretrain) # load the model
    pre_train_model = nn.Sequential(*list(rnn_pretrain_model.children())[:1])  # only take the LSTM part

    print(pre_train_model)
