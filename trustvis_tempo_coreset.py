########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
sys.path.append('..')
import os
import json
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import VisModel
from singleVis.losses import UmapLoss, ReconstructionLoss, DVILoss, DummyTemporalLoss
from singleVis.edge_dataset import VisDataHandler
from singleVis.trainer import VISTrainer
from singleVis.eval.evaluator import Evaluator
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import Trustvis_SpatialEdgeConstructor, TrustvisTemporalSpatialEdgeConstructor, TrustvisBorderSpatialEdgeConstructor
# from singleVis.spatial_skeleton_edge_constructor import ProxyBasedSpatialEdgeConstructor

from singleVis.projector import VISProjector
from singleVis.utils import find_neighbor_preserving_rate
from sklearn.neighbors import NearestNeighbors
from singleVis.coreset_selection import euclidean_dist_pair_np,LazyGreedy,FacilityLocation
import torch.nn.functional as F
from pynndescent import NNDescent
from sklearn.utils import check_random_state
########################################################################################################################
#                                                      PARAMETERS                                                   #
########################################################################################################################
"""This serve as an example of DeepVisualInsight implementation in pytorch."""
VIS_METHOD = "DVI" # DeepVisualInsight

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')

# get workspace dir
current_path = os.getcwd()

parent_path = os.path.dirname(current_path)

new_path = os.path.join(parent_path, 'training_dynamic')


parser.add_argument('--content_path', type=str)
parser.add_argument('--start', type=int,default=1)
parser.add_argument('--end', type=int,default=2)
# parser.add_argument('--epoch' , type=int)
parser.add_argument('--pred' , type=float, default=0.5)

# parser.add_argument('--epoch_end', type=int)
parser.add_argument('--epoch_period', type=int,default=1)
parser.add_argument('--preprocess', type=int,default=0)
parser.add_argument('--base',type=bool,default=False)
args = parser.parse_args()


CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config[VIS_METHOD]

# record output information
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]

EPOCH_START = args.start
EPOCH_END = args.end
# EPOCH_START = 1
# EPOCH_END = 50
EPOCH_PERIOD = args.epoch_period

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA1 = 1
LAMBDA2 = VISUALIZATION_PARAMETER["LAMBDA2"]
B_N_EPOCHS = 0
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]

S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]
VIS_MODEL_NAME = 'trustvis_tempo_coreset_selection' ### saved_as VIS_MODEL_NAME.pth


# Define hyperparameters
GPU_ID = 1
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
print("device", DEVICE)

def if_border(data):
    norm_preds = norm(data)

    sort_preds = np.sort(norm_preds, axis=1)
    diff = sort_preds[:, -1] - sort_preds[:, -2]
    border = np.zeros(len(diff), dtype=np.uint8) + 0.05
    border[diff < 0.15] = 1
        
    return border

def critical_prediction_flip(ref_pred, tar_pred):
    critical_prediction_flip_list = []
    for i in range(len(ref_pred)):
        if ref_pred[i] != tar_pred[i]:
            critical_prediction_flip_list.append(i)
    return critical_prediction_flip_list
            
def critical_border_flip(ref_data, tar_data):
    critical_border_flip_list = []
    ref_border_list = if_border(ref_data)
    tar_border_list = if_border(tar_data)
    for i in range(len(ref_border_list)):
        if ref_border_list[i] != tar_border_list[i]:
            critical_border_flip_list.append(i)
    return critical_border_flip_list

def norm(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def vis_error(vis_error_list, pred, inv_pred):
    for i in range(len(pred)):
        if pred[i] != inv_pred[i]:
            vis_error_list.append(i)
    return vis_error_list

def critical_npr(ref_train_data, tar_train_data, N_NEIGHBORS):
    npr = find_neighbor_preserving_rate(ref_train_data, tar_train_data, N_NEIGHBORS)
    k_npr = int(len(npr) * 0.005)
    _, npr_low_indices = torch.topk(torch.from_numpy(npr).to(device=DEVICE), k_npr, largest=False)

    return npr_low_indices.tolist()
    

def critical_inv_sim(pred_origin, inv_pred_origin, DEVICE):
    inv_similarity = F.cosine_similarity(torch.from_numpy(pred_origin).to(device=DEVICE), torch.from_numpy(inv_pred_origin).to(device=DEVICE))
    k_err = int(len(inv_similarity) * 0.005)
    _, inv_low_indices = torch.topk(inv_similarity, k_err, largest=False)

    return inv_low_indices.tolist()

def _construct_fuzzy_complex(train_data, metric="euclidean"):
    # """
    # construct a vietoris-rips complex
    # """
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
    # distance metric
    # # get nearest neighbors
    
    nnd = NNDescent(
        train_data,
        n_neighbors=15,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    knn_indices, knn_dists = nnd.neighbor_graph
    return knn_indices

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
# Define data_provider
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)
PREPROCESS = args.preprocess
if PREPROCESS:
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._gen_boundary(LEN // 10)    

# Define visualization models
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

#####  load exsiting vis model for transfer learning
# save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(EPOCH_START), ORIGIN_VIS_MODEL_NAME + ".pth")
# save_model = torch.load(save_dir, map_location="cpu")
# model.load_state_dict(save_model["state_dict"])
# model.to(DEVICE)
# model.eval()


# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)

# Define Projector
projector = VISProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)


start_flag = 1

prev_model = VisModel(ENCODER_DIMS, DECODER_DIMS)

# for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
#     temporal_k = 2
#     coreset_flag = 1
#     while temporal_k != 0:
#         # Define DVI Loss
#         if start_flag:
#             temporal_loss_fn = DummyTemporalLoss(DEVICE)
#             recon_loss_fn = ReconstructionLoss(beta=1.0)
#             umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, data_provider, iteration,net, 100, _a, _b,  repulsion_strength=1.0)
#             criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)

#             t0 = time.time()
#             spatial_cons = Trustvis_SpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net)
#             edge_to, edge_from, probs, pred_probs, feature_vectors, attention = spatial_cons.construct()
#             t1 = time.time()
#             start_flag = 0
#             optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
#             lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
#             temporal_k = 0
#             coreset_flag = 0
#         else:
#             if coreset_flag:
#                 temporal_loss_fn = DummyTemporalLoss(DEVICE)
#                 recon_loss_fn = ReconstructionLoss(beta=1.0)
#                 umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, data_provider, iteration,net, 100, _a, _b,  repulsion_strength=1.0)
#                 criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
#                 t0 = time.time()
#                 spatial_cons = Trustvis_SpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net)
#                 edge_to, edge_from, probs, pred_probs, feature_vectors, attention = spatial_cons.construct()
#                 t1 = time.time()
#                 optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
#                 lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
#             else:
#                 recon_loss_fn = ReconstructionLoss(beta=1.0)
#                 umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, data_provider, iteration,net, 100, _a, _b,  repulsion_strength=1.0)
#                 # representation for last epoch
#                 ref_train_data = data_provider.train_representation(iteration-1).squeeze()
#                 pred_origin = data_provider.get_pred(iteration-1, ref_train_data)
#                 pred = pred_origin.argmax(axis=1)
#                 embedding_ref = projector.batch_project(iteration-1, ref_train_data)
#                 inv_ref_data = projector.batch_inverse(iteration-1, embedding_ref)
#                 inv_pred_origin = data_provider.get_pred(iteration-1, inv_ref_data)
#                 inv_pred = inv_pred_origin.argmax(axis=1)  

#                 # representation for current epoch
#                 tar_train_data = data_provider.train_representation(iteration).squeeze()
#                 embedding_tar = projector.batch_project(iteration-1, tar_train_data)
#                 inv_tar_data = projector.batch_inverse(iteration-1, embedding_tar)
#                 new_pred_origin = data_provider.get_pred(iteration, tar_train_data)
#                 new_pred = new_pred_origin.argmax(axis=1)
#                 inv_new_pred_origin = data_provider.get_pred(iteration, inv_tar_data)
#                 inv_new_pred = inv_new_pred_origin.argmax(axis=1)
#                 embedding_tar_ = projector.batch_project(iteration, tar_train_data)
#                 inv_tar_data_ = projector.batch_inverse(iteration, embedding_tar_)
#                 inv_new_pred_origin_ = data_provider.get_pred(iteration, inv_tar_data_)
#                 inv_new_pred_ = inv_new_pred_origin_.argmax(axis=1)
                
#                 if temporal_k == 2:
#                     vis_error_list = []
#                     vis_error_list = vis_error(vis_error_list, pred, inv_pred)
#                     vis_error_list = vis_error(vis_error_list, pred, inv_new_pred)

#                     high_dim_prediction_flip_list = critical_prediction_flip(pred, new_pred)
#                     high_dim_border_flip_list = critical_border_flip(pred_origin, new_pred_origin)

#                     critical_set = set(high_dim_prediction_flip_list).union(set(high_dim_border_flip_list))
#                     critical_list = list(critical_set.union(set(vis_error_list)))

#                 else:
#                     vis_error_list = []
#                     vis_error_list = vis_error(vis_error_list, new_pred, inv_new_pred)
#                     vis_error_list = vis_error(vis_error_list, new_pred, inv_new_pred_)

#                 npr_low_indices = critical_npr(ref_train_data, tar_train_data, N_NEIGHBORS)
#                 inv_low_indices = critical_inv_sim(pred_origin, inv_pred_origin, DEVICE)

#                 critical_list = list(set(critical_list).union(set(npr_low_indices)))
#                 critical_list = list(set(critical_list).union(set(inv_low_indices)))

#                 knn_indices = _construct_fuzzy_complex(new_pred_origin)
#                 knn_indices = knn_indices[critical_list]

#                 knn_indices_flat = knn_indices.flatten()

#                 diff_list = list(set(knn_indices_flat).union(set(critical_list)))

#                 diff_list = list(set(diff_list).union(set(selection_result)))
#                 diff_data = tar_train_data[diff_list]

#                 ##### construct the spitial complex
#                 t0 = time.time()
#                 spatial_cons = TrustvisTemporalSpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net, diff_data=diff_data, sim_data=diff_data)
#                 edge_to, edge_from, probs, pred_probs, feature_vectors, attention, knn_indices = spatial_cons.construct()

#                 # Define training parameters
#                 temporal_loss_fn = DummyTemporalLoss(DEVICE)
#                 recon_loss_fn = ReconstructionLoss(beta=1.0)
#                 criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=3*LAMBDA1, lambd2=0.0,device=DEVICE)
#                 optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-5)
#                 lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
#                 temporal_k -= 1
#                 t_1= time.time()


#         print('complex-construct:', t1-t0)

#         pred_list = data_provider.get_pred(iteration, feature_vectors)
#         dataset = VisDataHandler(edge_to, edge_from, feature_vectors, attention, pred_probs,pred_list)
        
#         n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
#         # chose sampler based on the number of dataset
#         if len(edge_to) > pow(2,24):
#             sampler = CustomWeightedRandomSampler(pred_probs, n_samples, replacement=True)
#         else:
#             sampler = WeightedRandomSampler(pred_probs, n_samples, replacement=True)

#         edge_loader = DataLoader(dataset, batch_size=2000, sampler=sampler, num_workers=8, prefetch_factor=10)
        

#         ########################################################################################################################
#         #                                                       TRAIN                                                          #
#         ########################################################################################################################

#         # trainer = DVITrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)
        
#         trainer = VISTrainer(model,criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

#         t2=time.time()
#         if coreset_flag:
#             core_epoch = 1
#             trainer.train(PATIENT, core_epoch,data_provider,iteration)
#             grad = trainer.calc_gradient(data_provider,iteration,core_epoch).cpu().numpy()
#             train_labels = data_provider.train_labels(iteration)
#             train_num = len(data_provider.train_representation(iteration).squeeze())
#             all_index = list(range(train_num))
#             num_classes = 10
#             selection_result = np.array([], dtype=np.int32)
#             for c in range(num_classes):
#                 class_index = np.arange(train_num)[train_labels == c]
#                 gradients_class = euclidean_dist_pair_np(grad[class_index])
#                 matrix = -1. * gradients_class
#                 matrix -= np.min(matrix) - 1e-6
#                 all_index = np.array(range(len(class_index)))
#                 submod_function = FacilityLocation(index=all_index, similarity_matrix=matrix)
#                 submod_optimizer = LazyGreedy(index=all_index, budget=int(0.1*(len(class_index))))
#                 class_result = submod_optimizer.select(gain_function=submod_function.calc_gain,
#                                                         update_state=submod_function.update_state)
#                 selection_result = np.append(selection_result, class_index[class_result]).tolist()
#             coreset_flag = 0
#         else:
#             trainer.train(PATIENT, MAX_EPOCH,data_provider,iteration)
#         # trainer.train(PATIENT, MAX_EPOCH)
#         t3 = time.time()
#         print('training:', t3-t2)
#         # save result
#         save_dir = data_provider.model_path
#         trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "complex_construction", str(iteration), t1-t0)
#         trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "training", str(iteration), t3-t2)
#         save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
#         trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

#         print("Finish epoch {}...".format(iteration))

#         prev_model.load_state_dict(model.state_dict())
#         for param in prev_model.parameters():
#             param.requires_grad = False
#         w_prev = dict(prev_model.named_parameters())
        
# for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
#     train_data = data_provider.train_representation(iteration)
#     train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
#     emb = projector.batch_project(iteration, train_data)
#     inv = projector.batch_inverse(iteration, emb)
#     save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
#     train_data_loc = os.path.join(save_dir, "embedding.npy")
    # np.save(train_data_loc, emb)
    # inv_loc = os.path.join(save_dir, "inv.npy")
    # np.save(inv_loc, inv)
    # cluster_rep_loc = os.path.join(save_dir, "cluster_centers.npy")
    # cluster_rep = np.load(cluster_rep_loc)
    # emb = projector.batch_project(iteration, cluster_rep)
    # inv = projector.batch_inverse(iteration, emb)
    # inv_loc = os.path.join(save_dir, "inv_cluster.npy")
    # np.save(inv_loc, inv)
########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

# from singleVis.visualizer import visualizer
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# vis = visualizer(data_provider, projector, 200, "tab10")
# save_dir = os.path.join(data_provider.content_path, VIS_MODEL_NAME)

# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     vis.savefig(i, path=os.path.join(save_dir, "{}_{}_{}_{}.png".format(DATASET, i, VIS_METHOD,now)))
#     vis.get_background(i, 200)

# emb = projector.batch_project(data_provider)

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
evaluator = Evaluator(data_provider, projector)


# Evaluation_NAME = '{}_eval'.format(VIS_MODEL_NAME)
# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     evaluator.save_epoch_eval(i, 15, temporal_k=5, file_name="{}".format(Evaluation_NAME))

temporal_train = evaluator.eval_temporal_local_corr_train(2,1)
temporal_test = evaluator.eval_temporal_local_corr_test(2,1)
print(temporal_train)
print(temporal_test)
