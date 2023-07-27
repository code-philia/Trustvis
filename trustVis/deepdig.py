import torch
import torch.nn as nn
import torch.optim as optim
import os

content_path = "/home/yifan/dataset/clean/pairflip/cifar10/0"
model_loc = os.path.join(content_path,'Model','Epoch_{}'.format(60), 'subject_model.pth')
