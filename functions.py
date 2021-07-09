import torch
import torch.nn as nn
import numpy as np

import kaldiark
from apc import toy_lstm
import glob

# Model loading
def load_model(path, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
