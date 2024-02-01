####### dropout resnet18 vs without dropout
#### 
import copy
import torch
import sys
# sys.path.append("/home/yiming/GridSkeletonVis")
import numpy as np
import os, json

from singleVis.SingleVisualizationModel import VisModel
from singleVis.utils import *
from singleVis.projector import TimeVisProjector
from singleVis.eval.evaluate import *
import torch
import json

from scipy import stats as stats
from singleVis.projector import DVIProjector

import argparse
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('--start', type=int,default=1)
parser.add_argument('--end', type=int,default=3)
parser.add_argument('--epoch' , type=int,default=100)
args = parser.parse_args()
# CONTENT_PATH = "/home/yiming/EXP/CIFAR10_Clean"
CONTENT_PATH = args.content_path

# CONTENT_PATH = "/home/yiming/ContrastDebugger"
sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config['DVI']

# record output information
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
# REF_EPOCH = args.epoch
# TAR_EPOCH = REF_EPOCH + 1
EPOCH_START = args.start
EPOCH_END = args.end
EPOCH_PERIOD = config["EPOCH_PERIOD"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
# LAMBDA1 = VISUALIZATION_PARAMETER["LAMBDA1"]
# LAMBDA2 = VISUALIZATION_PARAMETER["LAMBDA2"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

# VIS_MODEL_NAME = 'trustvis_sk'
# VIS_MODEL_NAME = 'dvi'
# TAR_VIS_MODEL_NAME = 'dvi_overfit_all_transfer'
# TAR_VIS_MODEL_NAME = 'trustbase_gen_border_conf_margin_new'
# VIS_MODEL_NAME = 'trustbase_gen_border_conf_margin_new'
# TAR_VIS_MODEL_NAME = 'trustbase_gen_border'
# VIS_MODEL_NAME = 'trustbase_gen_border'
VIS_MODEL_NAME = 'trustvis_modi'
# VIS_MODEL_NAME = 'trustvis_tempo'
# TAR_VIS_MODEL_NAME = 'dvi_probs_pred'
# VIS_MODEL_NAME = 'dvi_probs_pred'
# TAR_VIS_MODEL_NAME = 'dvi'
VIS_MODEL_NAME = 'dvi_eval'
# VIS_MODEL_NAME = 'vis'
# VIS_MODEL_NAME = 'trustvis_remove_sampling'
# VIS_MODEL_NAME = 'trustvis_tempo_ablation'
# VIS_MODEL_NAME = 'base_dvi'
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

model = VisModel(ENCODER_DIMS, DECODER_DIMS)

projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
# projector = TimeVisProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)


n_neighbors = 15
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))
from singleVis.data import NormalDataProvider
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, 1, device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)

from singleVis.eval.evaluator import Evaluator
evaluator = Evaluator(data_provider, projector)

evaluation = dict()
if "b_train" not in evaluation:
    evaluation["b_train"] = dict()
if "b_test" not in evaluation:
    evaluation["b_test"] = dict()

for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    evaluation["b_train"][i] = evaluator.eval_b_train(i, 15)
    evaluation["b_test"][i] = evaluator.eval_b_test(i, 15)

Evaluation_NAME = '{}_eval'.format(VIS_MODEL_NAME)
file_name="{}".format(Evaluation_NAME)
save_dir = os.path.join(data_provider.model_path)
save_file = os.path.join(save_dir, file_name + ".json")
with open(save_file, "w") as f:
    json.dump(evaluation, f)

# if "vis_error_train" not in evaluation:
#     evaluation["vis_error_train"] = dict()
# if "vis_error_test" not in evaluation:
#     evaluation["vis_error_test"] = dict()

# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     _, vis_error_train = evaluator.eval_inv_train(i)
#     _, vis_error_test = evaluator.eval_inv_test(i)
#     evaluation["vis_error_train"][i] = int(vis_error_train)
#     evaluation["vis_error_test"][i] = int(vis_error_test)

# Evaluation_NAME = '{}_eval'.format(VIS_MODEL_NAME)
# file_name="{}".format(Evaluation_NAME)
# save_dir = os.path.join(data_provider.model_path)
# save_file = os.path.join(save_dir, file_name + ".json")
# with open(save_file, "w") as f:
#     json.dump(evaluation, f)