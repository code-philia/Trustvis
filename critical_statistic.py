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

# def if_border(data):
#     # mesh_preds = data_provider.get_pred(REF_EPOCH, data)
#     mesh_preds = data + 1e-8

#     sort_preds = np.sort(mesh_preds, axis=1)
#     diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
#     border = np.zeros(len(diff), dtype=np.uint8) + 0.05
#     border[diff < 0.15] = 1
        
#     return border
import argparse
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--epoch' , type=int,default=100)
args = parser.parse_args()
CONTENT_PATH = "/home/yifan/0ExpMinist/GoogleNet/01"
# CONTENT_PATH = "/home/yiming/EXP/CIFAR10_Clean"
# CONTENT_PATH = "/home/yiming/ContrastDebugger"
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
REF_EPOCH = args.epoch
TAR_EPOCH = REF_EPOCH + 1
EPOCH_START = REF_EPOCH
EPOCH_END = REF_EPOCH + 2
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
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]


# VIS_MODEL_NAME = 'trustvis_modi'
# VIS_MODEL_NAME = 'trustvis_repell'

# TAR_VIS_MODEL_NAME = 'dvi_eval'
VIS_MODEL_NAME = 'base_dvi'
# VIS_MODEL_NAME = 'vis'
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

model = VisModel(ENCODER_DIMS, DECODER_DIMS)

projector = VISProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
# projector = TimeVisProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)

n_neighbors = 15
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))
from singleVis.data import NormalDataProvider
data_provider = NormalDataProvider(CONTENT_PATH, net,EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch', classes=CLASSES,verbose=0)

def critical_prediction_flip(ref_pred, tar_pred):
    critical_prediction_flip_list = []
    critical_prediction_flip_from_list = []
    critical_prediction_flip_to_list = []
    for i in range(len(ref_pred)):
        if ref_pred[i] != tar_pred[i]:
            critical_prediction_flip_list.append(i)
            critical_prediction_flip_from_list.append(ref_pred[i])
            critical_prediction_flip_to_list.append(tar_pred[i])
    return critical_prediction_flip_list, critical_prediction_flip_from_list, critical_prediction_flip_to_list
            
def critical_border_flip(ref_data, tar_data):
    critical_border_flip_list = []
    critical_border_flip_from_list = []
    critical_border_flip_to_list = []
    ref_pred = ref_data.argmax(axis=1)
    tar_pred = tar_data.argmax(axis=1)

    # ref_border_list_2 = is_border(ref_data)
    # tar_border_list_2 = is_border(tar_data)
    ref_border_list = if_border(ref_data)
    tar_border_list = if_border(tar_data)
    ref_border_list_value = []
    tar_border_list_value = []
    for i in range(len(ref_border_list)):
        if ref_border_list[i] != tar_border_list[i]:
            critical_border_flip_list.append(i)
            ref_border_list_value.append(ref_border_list[i])
            tar_border_list_value.append(tar_border_list[i])
            critical_border_flip_from_list.append(ref_pred[i])
            critical_border_flip_to_list.append(tar_pred[i])
    return critical_border_flip_list, ref_border_list_value, tar_border_list_value, ref_border_list, tar_border_list, critical_border_flip_from_list, critical_border_flip_to_list

def norm(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

ref_train_data = data_provider.train_representation(REF_EPOCH).squeeze()
# ref_train_data = ref_train_data.reshape(ref_train_data.shape[0],ref_train_data.shape[1])

tar_train_data = data_provider.train_representation(TAR_EPOCH).squeeze()
# tar_train_data = ref_train_data.reshape(tar_train_data.shape[0],tar_train_data.shape[1])

# select similar samples as proxies
embedding_ref = projector.batch_project(REF_EPOCH, ref_train_data)
inv_ref_data = projector.batch_inverse(REF_EPOCH, embedding_ref)
# inv_ref_data = inv_ref_data.reshape(inv_ref_data.shape[0],ref_train_data.shape[1])
pred_origin = data_provider.get_pred(REF_EPOCH, ref_train_data)
pred = pred_origin.argmax(axis=1)
inv_pred_origin = data_provider.get_pred(REF_EPOCH, inv_ref_data)
inv_pred = inv_pred_origin.argmax(axis=1)


embedding_tar = projector.batch_project(TAR_EPOCH, tar_train_data)
inv_tar_data = projector.batch_inverse(TAR_EPOCH, embedding_tar)
# inv_tar_data = inv_tar_data.reshape(inv_tar_data.shape[0],inv_tar_data.shape[1])


new_pred_origin = data_provider.get_pred(TAR_EPOCH, tar_train_data)
inv_new_pred_origin = data_provider.get_pred(TAR_EPOCH, inv_tar_data)
new_pred = new_pred_origin.argmax(axis=1)
inv_new_pred = inv_new_pred_origin.argmax(axis=1)

high_dim_prediction_flip_list, high_critical_prediction_flip_from_list, high_critical_prediction_flip_to_list = critical_prediction_flip(pred, new_pred)
low_dim_prediction_flip_list, low_critical_prediction_flip_from_list, low_critical_prediction_flip_to_list = critical_prediction_flip(inv_pred, inv_new_pred)

high_dim_border_flip_list, ref_border_list_value, tar_border_list_value,high_ref_border_list,high_tar_border_list, critical_border_flip_from_list, critical_border_flip_to_list = critical_border_flip(pred_origin, new_pred_origin)
low_dim_border_flip_list, low_ref_border_list_value, low_tar_border_list_value, ref_border_list, tar_border_list, low_critical_border_flip_from_list, low_critical_border_flip_to_list = critical_border_flip(inv_pred_origin, inv_new_pred_origin)


ref_vis_error_list = []
for i in range(len(pred)):
    if pred[i] != inv_pred[i]:
        ref_vis_error_list.append(i)

tar_vis_error_list = []
for i in range(len(new_pred)):
    if new_pred[i] != inv_new_pred[i]:
        tar_vis_error_list.append(i)

# print(len(ref_vis_error_list))
# print(len(tar_vis_error_list))

embedding_tar_ = projector.batch_project(REF_EPOCH, tar_train_data)

inv_tar_data_ = projector.batch_inverse(REF_EPOCH, embedding_tar_)
inv_tar_data_ = inv_tar_data_.reshape(inv_tar_data_.shape[0],inv_tar_data_.shape[1])
inv_new_pred_ = data_provider.get_pred(TAR_EPOCH, inv_tar_data_).argmax(axis=1)
tar_vis_error_list_ = []
for i in range(len(new_pred)):
    if new_pred[i] != inv_new_pred_[i]:
        tar_vis_error_list_.append(i)

prediction_flip = set(high_dim_prediction_flip_list).intersection(set(low_dim_prediction_flip_list))
overlap_list = set(high_dim_prediction_flip_list).intersection(set(high_dim_border_flip_list))
print("overlap", len(overlap_list))
true_prediction_flip = 0
for i in range(len(high_dim_prediction_flip_list)):
    for j in range(len(low_dim_prediction_flip_list)):
        if high_dim_prediction_flip_list[i] == low_dim_prediction_flip_list[j]:
            if high_critical_prediction_flip_from_list[i] == low_critical_prediction_flip_from_list[j]:
                if high_critical_prediction_flip_to_list[i] == low_critical_prediction_flip_to_list[j]:
                    true_prediction_flip += 1

if len(low_dim_prediction_flip_list) != 0:
    prediction_flip_precision = len(prediction_flip) / len(low_dim_prediction_flip_list)
else:
    prediction_flip_precision = 0

if len(high_dim_prediction_flip_list) != 0:
    prediction_flip_recall = len(prediction_flip) / len(high_dim_prediction_flip_list)
else:
    prediction_flip_recall = 0
# prediction_flip_recall = len(prediction_flip) / len(high_dim_prediction_flip_list)

border_flip = set(high_dim_border_flip_list).intersection(set(low_dim_border_flip_list))
if len(low_dim_border_flip_list) != 0:
    border_flip_precision = len(border_flip) / len(low_dim_border_flip_list)
else:
    border_flip_precision = 0
border_flip_recall = len(border_flip) / len(high_dim_border_flip_list)

true_border_flip = 0
for i in range(len(high_dim_border_flip_list)):
    for j in range(len(low_dim_border_flip_list)):
        if high_dim_border_flip_list[i] == low_dim_border_flip_list[j]:
            if critical_border_flip_from_list[i] == low_critical_border_flip_from_list[j]:
                if critical_border_flip_to_list[i] == low_critical_border_flip_to_list[j]:
                    true_border_flip += 1

# print(len(prediction_flip), len(low_dim_prediction_flip_list), len(high_dim_prediction_flip_list), prediction_flip_precision, prediction_flip_recall)
# print(len(border_flip), len(low_dim_border_flip_list), len(high_dim_border_flip_list), border_flip_precision, border_flip_recall)
                    
# print(true_prediction_flip, len(low_dim_prediction_flip_list), len(high_dim_prediction_flip_list))
print("training pred flip precision",true_prediction_flip / len(low_dim_prediction_flip_list))
print("training pred flip recall ",true_prediction_flip / len(high_dim_prediction_flip_list))
# print(true_border_flip, len(low_dim_border_flip_list), len(high_dim_border_flip_list))
print("training bon confidence precision", true_border_flip / len(low_dim_border_flip_list)),
print("training bon confidence recall ", true_border_flip / len(high_dim_border_flip_list))

critical_list = list(set(high_dim_prediction_flip_list).union(set(high_dim_border_flip_list)))
print(len(critical_list))
critical_list = list(set(critical_list).union(set(tar_vis_error_list_)))
critical_data = tar_train_data[critical_list]

k_neighbors = 15
high_neigh = NearestNeighbors(n_neighbors=k_neighbors, radius=0.4)
high_neigh.fit(tar_train_data)
knn_dists, knn_indices = high_neigh.kneighbors(critical_data, n_neighbors=k_neighbors, return_distance=True)
knn_indices_flat = knn_indices.flatten()
diff_list = list(set(knn_indices_flat).union(set(critical_list)))
print(len(diff_list))
print(len(tar_vis_error_list_))
difference_list = list(set(diff_list).union(set(ref_vis_error_list)))

similarity = F.cosine_similarity(torch.from_numpy(pred_origin).to(device=DEVICE), torch.from_numpy(new_pred_origin).to(device=DEVICE))
# 计算要找的值的数量（百分之一长度）
k = int(len(similarity) * 0.1)
# 使用 topk 函数找到最大的前 k 个值及其索引
top_values, top_indices = torch.topk(similarity, k)
top_indices = top_indices.tolist()


sim_list = list(set(top_indices) - set(difference_list))

# print("neighbor overlap", len(set(tar_vis_error_list).intersection(set(knn_indices_flat) - set(critical_list))))
# print("critical overlap", len(set(tar_vis_error_list).intersection(set(high_dim_prediction_flip_list).union(set(high_dim_border_flip_list)))))
# print("tar_vis_error_list_ overlap", len(set(tar_vis_error_list).intersection(set(tar_vis_error_list_))))
# # print("tar vis error critical overlap", len(set(tar_vis_error_list).intersection(set(ref_vis_error_list))))
# print("difference_list overlap", len(set(tar_vis_error_list).intersection(set(difference_list))))
# print("sim_list overlap", len(set(tar_vis_error_list).intersection(set(sim_list))))
# print("tar vis error critical overlap", len(set(ref_vis_error_list).intersection(set(tar_vis_error_list_))))
# print("tar vis error critical overlap", len((set(ref_vis_error_list).intersection(set(tar_vis_error_list_))).intersection(set(tar_vis_error_list))))
# print("tar vis error critical overlap", len(set(tar_vis_error_list) - set(difference_list)))

# print("ref vis error overlap", set(ref_vis_error_list).intersection(set(high_dim_border_flip_list)))
# print("tar vis error overlap", set(tar_vis_error_list).intersection(set(high_dim_border_flip_list)))
# if REF_EPOCH > 70:
#     print(prediction_flip)
#     print(high_dim_prediction_flip_list)
#     print(border_flip)
#     print(high_dim_border_flip_list)
#     print(ref_border_list_value)
#     print(tar_border_list_value)
#     print(ref_border_list[high_dim_border_flip_list])
#     print(tar_border_list[high_dim_border_flip_list])

ref_border_indices = [i for i, value in enumerate(high_ref_border_list) if value == 1.0]
# print(ref_border_indices)
ref_border_flip = list(set(ref_border_indices).intersection(set(high_dim_border_flip_list)))

critical_border_flip_train_data = ref_train_data[ref_border_flip]
critical_border_flip_pred = pred[ref_border_flip]

# from singleVis.visualizer import visualizer
# vis = visualizer(data_provider, projector, 200)
# save_dir = os.path.join(data_provider.content_path, "imgTempo_")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# print(len(high_dim_border_flip_list))
# # vis.savefig_cus(epoch, ref_train_data, pred, pred, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_MODEL_NAME, epoch, "191_tempo_grad")))
# # vis.savefig_cus(REF_EPOCH, critical_prediction_flip_train_data, critical_prediction_flip_pred, critical_prediction_flip_pred, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_MODEL_NAME, REF_EPOCH, "191_tempo_grad_critical_prediction_all")))
# vis.savefig_cus(REF_EPOCH, critical_border_flip_train_data, critical_border_flip_pred, critical_border_flip_pred, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_MODEL_NAME, REF_EPOCH, "border")))

ref_border_indices = [i for i, value in enumerate(high_tar_border_list) if value == 1.0]
# print(ref_border_indices)
ref_border_flip = list(set(ref_border_indices).intersection(set(high_dim_border_flip_list)))

# critical_border_flip_train_data = tar_train_data[ref_border_flip]
# critical_border_flip_pred = pred[ref_border_flip]
# vis.savefig_cus(TAR_EPOCH, critical_border_flip_train_data, critical_border_flip_pred, critical_border_flip_pred, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_MODEL_NAME, TAR_EPOCH, "border")))

testing_data  = data_provider.test_representation(TAR_EPOCH)
testing_data = testing_data.reshape(testing_data.shape[0],testing_data.shape[1])

testing_emd = projector.batch_project(TAR_EPOCH, testing_data)
testing_new_data = projector.batch_inverse(TAR_EPOCH, testing_emd)
testing_new_data = testing_new_data.reshape(testing_new_data.shape[0],testing_new_data.shape[1])

testpred_origin =  data_provider.get_pred(TAR_EPOCH, testing_data)
new_testpred_origin = data_provider.get_pred(TAR_EPOCH, testing_new_data)
testpred =  testpred_origin.argmax(axis=1)
new_testpred = new_testpred_origin.argmax(axis=1)

ref_testing_data  = data_provider.test_representation(REF_EPOCH)
ref_testing_data = ref_testing_data.reshape(ref_testing_data.shape[0],ref_testing_data.shape[1])

testing_emd = projector.batch_project(REF_EPOCH, ref_testing_data)
ref_testing_new_data = projector.batch_inverse(REF_EPOCH, testing_emd)
ref_testing_new_data = ref_testing_new_data.reshape(ref_testing_new_data.shape[0],ref_testing_new_data.shape[1])


ref_testpred_origin =  data_provider.get_pred(REF_EPOCH, ref_testing_data)
ref_new_testpred_origin = data_provider.get_pred(REF_EPOCH, ref_testing_new_data)
ref_testpred =  ref_testpred_origin.argmax(axis=1)
ref_new_testpred = ref_new_testpred_origin.argmax(axis=1)


ref_vis_error_list = []
for i in range(len(testpred)):
    if testpred[i] != new_testpred[i]:
        ref_vis_error_list.append(i)

tar_vis_error_list = []
for i in range(len(ref_testpred)):
    if ref_testpred[i] != ref_new_testpred[i]:
        tar_vis_error_list.append(i)

# print(len(ref_vis_error_list))
# file_path = '/home/yiming/trustvis/ref_vis_error.json'
# with open(file_path, 'w') as file:
#     json.dump(tar_vis_error_list, file)
# print(len(tar_vis_error_list))

high_dim_prediction_flip_list, high_critical_prediction_flip_from_list, high_critical_prediction_flip_to_list = critical_prediction_flip(ref_testpred, testpred)
low_dim_prediction_flip_list, low_critical_prediction_flip_from_list, low_critical_prediction_flip_to_list= critical_prediction_flip(ref_new_testpred, new_testpred)

high_dim_border_flip_list, ref_border_list_value, tar_border_list_value,_,_, critical_border_flip_from_list, critical_border_flip_to_list = critical_border_flip(ref_testpred_origin, testpred_origin)
low_dim_border_flip_list, low_ref_border_list_value, low_tar_border_list_value,_,_, low_critical_border_flip_from_list, low_critical_border_flip_to_list = critical_border_flip(ref_new_testpred_origin, new_testpred_origin)

prediction_flip = set(high_dim_prediction_flip_list).intersection(set(low_dim_prediction_flip_list))
prediction_flip_precision = len(prediction_flip) / len(low_dim_prediction_flip_list)
prediction_flip_recall = len(prediction_flip) / len(high_dim_prediction_flip_list)

true_prediction_flip = 0
for i in range(len(high_dim_prediction_flip_list)):
    for j in range(len(low_dim_prediction_flip_list)):
        if high_dim_prediction_flip_list[i] == low_dim_prediction_flip_list[j]:
            if high_critical_prediction_flip_from_list[i] == low_critical_prediction_flip_from_list[j]:
                if high_critical_prediction_flip_to_list[i] == low_critical_prediction_flip_to_list[j]:
                    true_prediction_flip += 1

border_flip = set(high_dim_border_flip_list).intersection(set(low_dim_border_flip_list))
if len(low_dim_border_flip_list) != 0:
    border_flip_precision = len(border_flip) / len(low_dim_border_flip_list)
else:
    border_flip_precision = 0
border_flip_recall = len(border_flip) / len(high_dim_border_flip_list)

true_border_flip = 0
for i in range(len(high_dim_border_flip_list)):
    for j in range(len(low_dim_border_flip_list)):
        if high_dim_border_flip_list[i] == low_dim_border_flip_list[j]:
            if critical_border_flip_from_list[i] == low_critical_border_flip_from_list[j]:
                if critical_border_flip_to_list[i] == low_critical_border_flip_to_list[j]:
                    true_border_flip += 1

# print(len(prediction_flip), len(low_dim_prediction_flip_list), len(high_dim_prediction_flip_list), prediction_flip_precision, prediction_flip_recall)
# print(len(border_flip), len(low_dim_border_flip_list), len(high_dim_border_flip_list), border_flip_precision, border_flip_recall)
# print(true_prediction_flip, len(low_dim_prediction_flip_list), len(high_dim_prediction_flip_list)) 
print("testing precision", true_prediction_flip / len(low_dim_prediction_flip_list))  
print("testing recall", true_prediction_flip / len(high_dim_prediction_flip_list))
# print(true_border_flip, len(low_dim_border_flip_list), len(high_dim_border_flip_list))
print("testing bon confidence precision",  true_border_flip / len(low_dim_border_flip_list))
print("testing bon confidence recall", true_border_flip / len(high_dim_border_flip_list))



# from singleVis.eval.evaluator import Evaluator
# evaluator = Evaluator(data_provider, projector)
# n = evaluator.eval_nn_train(REF_EPOCH, 15)
# n_1 = evaluator.eval_nn_test(REF_EPOCH, 15)

# p = evaluator.eval_inv_train(REF_EPOCH)
# p_1 = evaluator.eval_inv_test(REF_EPOCH)

# # t = evaluator.eval_temporal_local_corr_train(TAR_EPOCH, 1, EPOCH_START, EPOCH_END, EPOCH_PERIOD)
# # t_1 = evaluator.eval_temporal_local_corr_test(TAR_EPOCH, 1, EPOCH_START, EPOCH_END, EPOCH_PERIOD)

# # tn = evaluator.eval_temporal_train(15)
# # tn_1 = evaluator.eval_temporal_test(15)


# # n_2 = evaluator.eval_b_train(TAR_EPOCH, 15)
# # n_3 = evaluator.eval_b_test(TAR_EPOCH, 15)

# print(n, n_1)
# print(p, p_1)
# print(t, t_1)
# print(tn, tn_1)
# print(n_2, n_3)


# n = evaluator.eval_nn_train(REF_EPOCH, 15)
# n_1 = evaluator.eval_nn_test(REF_EPOCH, 15)

# p = evaluator.eval_inv_train(REF_EPOCH)
# p_1 = evaluator.eval_inv_test(REF_EPOCH)

# # n_2 = evaluator.eval_b_train(REF_EPOCH, 15)
# # n_3 = evaluator.eval_b_test(REF_EPOCH, 15)

# print(n, n_1)
# print(p, p_1)
# # print(n_2, n_3)
