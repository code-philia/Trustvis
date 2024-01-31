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
import torch.nn.functional as F
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
parser.add_argument('--end', type=int,default=3)
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
VIS_MODEL_NAME = 'trustvis_tempo' ### saved_as VIS_MODEL_NAME.pth


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

from pynndescent import NNDescent
from sklearn.utils import check_random_state

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

for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    temporal_k = 2
    while temporal_k != 0:
        # Define DVI Loss
        if start_flag:
            temporal_loss_fn = DummyTemporalLoss(DEVICE)
            # recon_loss_fn = ReconstructionPredLoss(data_provider=data_provider,epoch=iteration, beta=1.0)
            recon_loss_fn = ReconstructionLoss(beta=1.0)
            umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, data_provider, iteration,net, 100, _a, _b,  repulsion_strength=1.0)
            # umap_loss_fn = SementicUmapLoss(negative_sample_rate, DEVICE, data_provider, iteration, _a, _b, repulsion_strength=1.0)
            # recon_loss_fn = ReconstructionPredEdgeLoss(data_provider=data_provider,iteration=iteration, beta=1.0)
            criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
            # ref_train_data = data_provider.train_representation(iteration).squeeze()
            # ref_train_data = ref_train_data.reshape(ref_train_data.shape[0],ref_train_data.shape[1])
            # k_neighbors = 15
            # high_neigh = NearestNeighbors(n_neighbors=k_neighbors, radius=0.4)
            # high_neigh.fit(ref_train_data)
            # knn_dists, knn_indices = high_neigh.kneighbors(ref_train_data, n_neighbors=k_neighbors, return_distance=True)

            # pred_dif_list = []
            # pred_dif_index_list = []
            # gen_border_data = np.array([])
            # import random
            # pred_origin = data_provider.get_pred(iteration, ref_train_data)
            # pred_res = data_provider.get_pred(iteration, ref_train_data).argmax(axis=1)

            # for i in range(len(knn_indices)):
            # # for i in range(5000):
            #     neighbor_list = list(knn_indices[i])
            #     neighbor_data = ref_train_data[neighbor_list]
            #     # neighbor_pred_origin = pred_origin[neighbor_list]
            #     neighbor_pred = pred_res[neighbor_list]
            #     for j in range(len(neighbor_pred)):
            #         if neighbor_pred[0] != neighbor_pred[j]:
            #             # if iteration < ((EPOCH_END - EPOCH_START)*0.3):
            #             if iteration < 70:
            #                 random_number = random.randint(1, 7)
            #             else:
            #                 random_number = 1
            #             if random_number == 1:
            #             # gen_points = np.linspace(neighbor_data[0], neighbor_data[j], 3)[1:-1]
            #                 gen_points = np.array([(neighbor_data[0] + neighbor_data[j]) / 2])
            #                 if len(gen_border_data) > 0:
            #                     gen_border_data = np.concatenate((gen_border_data, gen_points), axis=0)
            #                 else:
            #                     gen_border_data = gen_points
            #                     print(gen_border_data.shape)

            #     if (i % 5000) == 0:
            #         print(i)

            # print(gen_border_data.shape)
            # sub_n = 10000
            # if len(gen_border_data) > 10000:
            #     random_indices = np.random.choice(len(gen_border_data), sub_n, replace=False)
            #     # random get subsets
            #     fin_gen_border_data = gen_border_data[random_indices, :]
            # else:
            #     fin_gen_border_data = gen_border_data
            t0 = time.time()
            # spatial_cons = Trustvis_SpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net)
            spatial_cons = TrustvisBorderSpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net, fin_gen_border_data)
            # spatial_cons = Trustvis_SpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net)
            edge_to, edge_from, probs, pred_probs, feature_vectors, attention = spatial_cons.construct()
            t1 = time.time()
            start_flag = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
            temporal_k = 0
        else:
            recon_loss_fn = ReconstructionLoss(beta=1.0)
            umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, data_provider, iteration,net, 100, _a, _b,  repulsion_strength=1.0)
            
            if temporal_k == 2:
                ref_train_data = data_provider.train_representation(iteration-1).squeeze()
                # ref_train_data = ref_train_data.reshape(ref_train_data.shape[0],ref_train_data.shape[1])
                tar_train_data = data_provider.train_representation(iteration).squeeze()
                # tar_train_data = ref_train_data.reshape(tar_train_data.shape[0],tar_train_data.shape[1])
                
                pred_origin = data_provider.get_pred(iteration-1, ref_train_data)
                pred = pred_origin.argmax(axis=1)

                embedding_ref = projector.batch_project(iteration-1, ref_train_data)
                inv_ref_data = projector.batch_inverse(iteration-1, embedding_ref)

                inv_pred_origin = data_provider.get_pred(iteration-1, inv_ref_data)
                inv_pred = inv_pred_origin.argmax(axis=1)

                vis_error_list = []
                for i in range(len(pred)):
                    if pred[i] != inv_pred[i]:
                        vis_error_list.append(i)

                embedding_tar = projector.batch_project(iteration-1, tar_train_data)
                inv_tar_data = projector.batch_inverse(iteration-1, embedding_tar)

                new_pred_origin = data_provider.get_pred(iteration, tar_train_data)
                new_pred = new_pred_origin.argmax(axis=1)

                inv_new_pred_origin = data_provider.get_pred(iteration, inv_tar_data)
                inv_new_pred = inv_new_pred_origin.argmax(axis=1)

                # vis_error_list = []
                for i in range(len(pred)):
                    if new_pred[i] != inv_new_pred[i]:
                        vis_error_list.append(i)

                high_dim_prediction_flip_list = critical_prediction_flip(pred, new_pred)
                high_dim_border_flip_list = critical_border_flip(pred_origin, new_pred_origin)
                

                critical_set = set(high_dim_prediction_flip_list).union(set(high_dim_border_flip_list))
                critical_list = list(critical_set.union(set(vis_error_list)))

                npr = find_neighbor_preserving_rate(ref_train_data, tar_train_data, N_NEIGHBORS)
                k_npr = int(len(npr) * 0.005)
                # 使用 topk 函数找到最小的前 k 个值及其索引
                npr_low_values, npr_low_indices = torch.topk(torch.from_numpy(npr).to(device=DEVICE), k_npr, largest=False)
                # npr_low_indices = torch.nonzero(torch.from_numpy(npr).to(device=DEVICE) <= 0.4, as_tuple=True)[0]

                inv_similarity = F.cosine_similarity(torch.from_numpy(pred_origin).to(device=DEVICE), torch.from_numpy(inv_pred_origin).to(device=DEVICE))
                # 计算要找的值的数量（百分之一长度）
                k_err = int(len(inv_similarity) * 0.005)
                # 使用 topk 函数找到最大的前 k 个值及其索引
                inv_low_values, inv_low_indices = torch.topk(inv_similarity, k_err, largest=False)
                # inv_low_indices = torch.nonzero(inv_similarity <= 0.2, as_tuple=True)[0]

                # critical_list = list(critical_set)
                critical_list = list(set(critical_list).union(set(npr_low_indices.tolist())))
                critical_list = list(set(critical_list).union(set(inv_low_indices.tolist())))
                critical_data = tar_train_data[critical_list]
                # print(len(critical_list))
            else:
                tar_train_data = data_provider.train_representation(iteration).squeeze()

                embedding_tar = projector.batch_project(iteration-1, tar_train_data)
                inv_tar_data = projector.batch_inverse(iteration-1, embedding_tar)

                new_pred_origin = data_provider.get_pred(iteration, tar_train_data)
                new_pred = new_pred_origin.argmax(axis=1)

                inv_new_pred_origin = data_provider.get_pred(iteration, inv_tar_data)
                inv_new_pred = inv_new_pred_origin.argmax(axis=1)

                embedding_tar_ = projector.batch_project(iteration, tar_train_data)
                inv_tar_data_ = projector.batch_inverse(iteration, embedding_tar_)

                inv_new_pred_origin_ = data_provider.get_pred(iteration, inv_tar_data_)
                inv_new_pred_ = inv_new_pred_origin_.argmax(axis=1)

                vis_error_list = []
                for i in range(len(new_pred)):
                    if new_pred[i] != inv_new_pred[i]:
                        vis_error_list.append(i)

                for i in range(len(new_pred)):
                    if new_pred[i] != inv_new_pred_[i]:
                        vis_error_list.append(i)

                # critical_list = list(critical_set.union(set(vis_error_list)))

                npr = find_neighbor_preserving_rate(ref_train_data, tar_train_data, N_NEIGHBORS)
                k_npr = int(len(npr) * 0.005)
                # 使用 topk 函数找到最小的前 k 个值及其索引
                npr_low_values, npr_low_indices = torch.topk(torch.from_numpy(npr).to(device=DEVICE), k_npr, largest=False)
                # npr_low_indices = torch.nonzero(torch.from_numpy(npr).to(device=DEVICE) <= 0.4, as_tuple=True)[0]

                inv_similarity = F.cosine_similarity(torch.from_numpy(new_pred_origin).to(device=DEVICE), torch.from_numpy(inv_new_pred_origin_).to(device=DEVICE))
                # 计算要找的值的数量（百分之一长度）
                k_err = int(len(inv_similarity) * 0.005)
                # 使用 topk 函数找到最大的前 k 个值及其索引
                inv_low_values, inv_low_indices = torch.topk(inv_similarity, k_err, largest=False)
                # inv_low_indices = torch.nonzero(inv_similarity <= 0.2, as_tuple=True)[0]

                # critical_list = list(critical_set)
                critical_list = list(set(vis_error_list).union(set(npr_low_indices.tolist())))
                critical_list = list(set(critical_list).union(set(inv_low_indices.tolist())))
                critical_data = tar_train_data[critical_list]
            similarity = F.cosine_similarity(torch.from_numpy(pred_origin).to(device=DEVICE), torch.from_numpy(new_pred_origin).to(device=DEVICE))
            # 计算要找的值的数量（百分之一长度）
            k = int(len(similarity) * 0.2)
            # 使用 topk 函数找到最大的前 k 个值及其索引
            top_values, top_indices = torch.topk(similarity, k)
            top_indices = top_indices.tolist()
            # final_critical_list = list(set(critical_list) - (set(top_indices)))

            # critical_data = tar_train_data[final_critical_list]

            k_neighbors = 15
            # high_neigh = NearestNeighbors(n_neighbors=k_neighbors, radius=0.4)
            # high_neigh.fit(tar_train_data)
            # knn_indices_ = _construct_fuzzy_complex(tar_train_data)
            knn_indices = _construct_fuzzy_complex(new_pred_origin)
            knn_indices = knn_indices[critical_list]
            # knn_indices_ = knn_indices_[critical_list]
            # knn_dists, knn_indices = high_neigh.kneighbors(critical_data, n_neighbors=k_neighbors, return_distance=True)
            knn_indices_flat = knn_indices.flatten()
            # knn_indices_flat_ = knn_indices_.flatten()
            # diff_set = set(knn_indices_flat).union(set(critical_list))
            diff_list = list(set(knn_indices_flat).union(set(critical_list)))
            # diff_list = list(set(diff_list).union(set(knn_indices_flat_)))

            # final_critical_list = list(set(diff_list) - (set(top_indices)))

            # diff_list = list(diff_set.union(set(vis_error_list)))
            # diff_list = list(set(knn_indices_flat))
            diff_data = tar_train_data[diff_list]
            filtered_data = [tar_train_data[i] for i in range(len(tar_train_data)) if i not in top_indices]

            # print(top_indices)
            final_list = list(set(diff_list).union(set(top_indices)))
            sim_data = tar_train_data[top_indices]

            # print(len(high_dim_prediction_flip_list))
            # print(len(high_dim_border_flip_list))
            # print(len(vis_error_list))
            ##### construct the spitial complex
            t0 = time.time()
            spatial_cons = TrustvisTemporalSpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net, diff_data=diff_data, sim_data=sim_data)
            t1 = time.time()
            edge_to, edge_from, probs, pred_probs, feature_vectors, attention, knn_indices = spatial_cons.construct()

            # Define training parameters
            temporal_loss_fn = DummyTemporalLoss(DEVICE)
            recon_loss_fn = ReconstructionLoss(beta=1.0)
            # recon_loss_fn = TemporalReconstructionLoss(beta=1.0)
            criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=3*LAMBDA1, lambd2=0.0,device=DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
            temporal_k -= 1
            t_1= time.time()
            # npr = torch.tensor(find_neighbor_preserving_rate(prev_data, curr_data, N_NEIGHBORS)).to(DEVICE)
            t_2= time.time()
        
            # temporal_loss_fn = TemporalLoss(w_prev, DEVICE)
            # criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr,device=DEVICE)
        # Define training parameters
        # optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
        # Define Edge dataset


        print('complex-construct:', t1-t0)



        pred_list = data_provider.get_pred(iteration, feature_vectors)
        dataset = VisDataHandler(edge_to, edge_from, feature_vectors, attention, pred_probs,pred_list)
        
        n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
        # chose sampler based on the number of dataset
        if len(edge_to) > pow(2,24):
            sampler = CustomWeightedRandomSampler(pred_probs, n_samples, replacement=True)
        else:
            sampler = WeightedRandomSampler(pred_probs, n_samples, replacement=True)

        edge_loader = DataLoader(dataset, batch_size=2000, sampler=sampler, num_workers=8, prefetch_factor=10)
        

        ########################################################################################################################
        #                                                       TRAIN                                                          #
        ########################################################################################################################

        # trainer = DVITrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)
        
        trainer = VISTrainer(model,criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

        t2=time.time()
        trainer.train(PATIENT, MAX_EPOCH,data_provider,iteration)
        # trainer.train(PATIENT, MAX_EPOCH)
        t3 = time.time()
        print('training:', t3-t2)
        # save result
        save_dir = data_provider.model_path
        trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "complex_construction", str(iteration), t1-t0)
        trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "training", str(iteration), t3-t2)
        save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
        trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

        print("Finish epoch {}...".format(iteration))

        prev_model.load_state_dict(model.state_dict())
        for param in prev_model.parameters():
            param.requires_grad = False
        w_prev = dict(prev_model.named_parameters())
        

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

from singleVis.visualizer import visualizer
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
vis = visualizer(data_provider, projector, 200, "tab10")
save_dir = os.path.join(data_provider.content_path, VIS_MODEL_NAME)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    vis.savefig(i, path=os.path.join(save_dir, "{}_{}_{}_{}.png".format(DATASET, i, VIS_METHOD,now)))

# emb = projector.batch_project(data_provider)

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
evaluator = Evaluator(data_provider, projector)


Evaluation_NAME = '{}_eval'.format(VIS_MODEL_NAME)
for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    evaluator.save_epoch_eval(i, 15, temporal_k=5, file_name="{}".format(Evaluation_NAME))
