####### dropout resnet18 vs without dropout
#### 
import copy
import torch
import sys
# sys.path.append('/home/yiming/trustvis')
import numpy as np
import os, json

import torch.optim as optim
# from pyemd import emd
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors

# sys.path.append("/home/yiming/ContrastDebugger")
from singleVis.SingleVisualizationModel import VisModel
from singleVis.utils import *
from singleVis.projector import TimeVisProjector, VISProjector
from singleVis.eval.evaluate import *
# from singleVis.utils import get_border_points
# from re import sub
import torch
# import math
# import tqdm
import json
import time
from pynndescent import NNDescent
# from sklearn.neighbors import KDTree
# from sklearn.metrics import pairwise_distances
from scipy import stats as stats
# from skeleton_generator_with_cluster import SkeletonGenerator,CenterSkeletonGenerator
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial import distance
# from scipy.stats import pearsonr

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# def if_border(data):
#     mesh_preds = data_provider.get_pred(REF_EPOCH, data)
#     mesh_preds = mesh_preds + 1e-8

#     sort_preds = np.sort(mesh_preds, axis=1)
#     diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
#     border = np.zeros(len(diff), dtype=np.uint8) + 0.05
#     border[diff < 0.15] = 1
        
#     return border

def if_border(data):
    norm_preds = norm(data)

    sort_preds = np.sort(norm_preds, axis=1)
    diff = sort_preds[:, -1] - sort_preds[:, -2]
    border = np.zeros(len(diff), dtype=np.uint8) + 0.05
    border[diff < 0.15] = 1
        
    return border

import argparse
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--vismodel' , type=str,default='')
parser.add_argument('--content_path' , type=str,default='')
parser.add_argument('--start',type=int)
parser.add_argument('--end',type=int)
args = parser.parse_args()
CONTENT_PATH = args.content_path

sys.path.append(CONTENT_PATH)
# config_dvi_modi.json
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config["DVI"]

# record output information
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
GPU_ID = 0

EPOCH_PERIOD = config["EPOCH_PERIOD"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA1 = 1
LAMBDA2 = 0
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]


VIS_MODEL_NAME = args.vismodel

EPOCH_START = args.start
EPOCH_END = args.end

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

model = VisModel(ENCODER_DIMS, DECODER_DIMS)

projector = VISProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
# projector = TimeVisProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)

sys.path.append(CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))
from singleVis.data import NormalDataProvider
data_provider = NormalDataProvider(CONTENT_PATH, net,EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch', classes=CLASSES,verbose=0)

from singleVis.eval.evaluator import Evaluator

data_provider = NormalDataProvider(CONTENT_PATH, net,EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch', classes=CLASSES,verbose=0)

evaluator = Evaluator(data_provider, projector)
evaluator.eval_critical_temporal_train(15)
evaluator.eval_critical_temporal_test(15)


train_val_corr, train_corr_std = evaluator.eval_critical_temporal_train(15)
test_val_corr, test_corr_std = evaluator.eval_critical_temporal_test(15)

print(train_val_corr,train_corr_std,test_val_corr,test_corr_std)
