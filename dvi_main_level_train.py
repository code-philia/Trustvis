########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
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
from singleVis.losses import UmapLoss, ReconstructionLoss, TemporalLoss, DVILoss, SingleVisLoss, DummyTemporalLoss
from singleVis.edge_dataset import DVIDataHandler
from singleVis.trainer import DVITrainer, DVIReFineTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import SingleEpochSpatialEdgeConstructor,SingleEpochSpatialEdgeConstructorLEVEL
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.utils import find_neighbor_preserving_rate
########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
"""This serve as an example of DeepVisualInsight implementation in pytorch."""
VIS_METHOD = "DVI" # DeepVisualInsight

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('--epoch', type=int)
# parser.add_argument('--epoch_end', type=int)
parser.add_argument('--epoch_period', type=int,default=1)
parser.add_argument('--preprocess', type=int,default=0)
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

EPOCH_START = args.epoch
EPOCH_END = args.epoch
EPOCH_PERIOD = args.epoch_period

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA1 = VISUALIZATION_PARAMETER["LAMBDA1"]
LAMBDA2 = VISUALIZATION_PARAMETER["LAMBDA2"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

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
        data_provider._estimate_boundary(LEN // 10, l_bound=L_BOUND)

# Define visualization models


ENCODER_DIMS_LIST = [[512,256,256,256,256,2],[2,2,2,2],[2,2,2,2,2]]
DECODER_DIMS_LIST = [[2,256,256,256,256,512],[2,2,2,2],[2,2,2,2,2]]

DIM_LIST = [256,128,2]



# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define Projector


prev_model_l = [VisModel(ENCODER_DIMS_LIST[0], DECODER_DIMS_LIST[0]), VisModel(ENCODER_DIMS_LIST[1], DECODER_DIMS_LIST[1]),VisModel(ENCODER_DIMS_LIST[2], DECODER_DIMS_LIST[2])]

projector1 = DVIProjector(vis_model=prev_model_l[0], content_path=CONTENT_PATH, vis_model_name='dvi_level1', device=DEVICE)
projector2 = DVIProjector(vis_model=prev_model_l[1], content_path=CONTENT_PATH, vis_model_name='dvi_level2', device=DEVICE)
projector3 = DVIProjector(vis_model=prev_model_l[2], content_path=CONTENT_PATH, vis_model_name='dvi_level3', device=DEVICE)
projects = [projector1,projector2,projector3]

start_flag = 1



for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):

    
    for i in range(len(ENCODER_DIMS_LIST)):
        CUR_DIM = DIM_LIST[i]
        print("start for level{}".format(i+1))
        model = VisModel(ENCODER_DIMS_LIST[i], DECODER_DIMS_LIST[i])
        project_l = projects[:i]
        # Define DVI Loss
        if start_flag:
            temporal_loss_fn = DummyTemporalLoss(DEVICE)
            criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
            start_flag = 0
        # else:
            # TODO AL mode, redefine train_representation
            # prev_data = data_provider.train_representation(iteration-EPOCH_PERIOD)
            # curr_data = data_provider.train_representation(iteration)
            # npr = torch.tensor(find_neighbor_preserving_rate(prev_data, curr_data, N_NEIGHBORS)).to(DEVICE)
            # temporal_loss_fn = TemporalLoss(w_prev, DEVICE)
            # criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr,device=DEVICE)
        # Define training parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
        # Define Edge dataset
        t0 = time.time()
        spatial_cons = SingleEpochSpatialEdgeConstructorLEVEL(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS,project_l,CUR_DIM)
        edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()
        t1 = time.time()

        probs = probs / (probs.max()+1e-3)
        eliminate_zeros = probs>5e-2    #1e-3
        edge_to = edge_to[eliminate_zeros]
        edge_from = edge_from[eliminate_zeros]
        probs = probs[eliminate_zeros]

        dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)

        n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
        # chose sampler based on the number of dataset
        if len(edge_to) > pow(2,24):
            sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
        else:
            sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
        edge_loader = DataLoader(dataset, batch_size=2000, sampler=sampler, num_workers=8, prefetch_factor=10)

        ########################################################################################################################
        #                                                       TRAIN                                                          #
        ########################################################################################################################

        trainer = DVITrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

        t2=time.time()
        trainer.train(PATIENT, MAX_EPOCH)
        t3 = time.time()

        # save result
        save_dir = data_provider.model_path
        trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "complex_construction", str(iteration), t1-t0)
        trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "training", str(iteration), t3-t2)
        save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
        trainer.save(save_dir=save_dir, file_name="{}_level{}".format(VIS_MODEL_NAME,i+1))

        print("Finish epoch {}...".format(iteration))

        # prev_model.load_state_dict(model.state_dict())
        # for param in prev_model.parameters():
        #     param.requires_grad = False
        # w_prev = dict(prev_model.named_parameters())
    

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

# from singleVis.visualizer import visualizer
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# vis = visualizer(data_provider, projector, 200, "tab10")
# save_dir = os.path.join(data_provider.content_path, "imgptDVI")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     vis.savefig(i, path=os.path.join(save_dir, "{}_{}_{}_{}.png".format(DATASET, i, VIS_METHOD,now)))

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
# eval_epochs = range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD)
# EVAL_EPOCH_DICT = {
#     "mnist":[1,10,15],
#     "fmnist":[1,25,50],
#     "cifar10":[1,100,199]
# }
# eval_epochs = EVAL_EPOCH_DICT[DATASET]
# evaluator = Evaluator(data_provider, projector)

# for eval_epoch in eval_epochs:
#     evaluator.save_epoch_eval(eval_epoch, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))
