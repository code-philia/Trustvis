########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
from torch import nn
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
from singleVis.losses import ReconstructionLoss, TemporalLoss, SingleVisLoss, DummyTemporalLoss
from singleVis.backend import convert_distance_to_probability, compute_cross_entropy
from singleVis.edge_dataset import VisDataHandler
from singleVis.trainer import BaseTextTrainer
from singleVis.eval.evaluator import Evaluator
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import SingleEpochTextSpatialEdgeConstructor
# from singleVis.spatial_skeleton_edge_constructor import ProxyBasedSpatialEdgeConstructor

from singleVis.projector import VISProjector
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

# get workspace dir
current_path = os.getcwd()

parent_path = os.path.dirname(current_path)

new_path = os.path.join(parent_path, 'training_dynamic')


parser.add_argument('--content_path', type=str,default=new_path)
parser.add_argument('--start', type=int,default=1)
parser.add_argument('--end', type=int,default=2)
parser.add_argument('--epoch' , type=int, default=3)

# parser.add_argument('--epoch_end', type=int)
parser.add_argument('--epoch_period', type=int,default=1)
parser.add_argument('--preprocess', type=int,default=0)
parser.add_argument('--base',type=bool,default=False)

# text args
# Required parameters
parser.add_argument("--output_dir", default="/home/yiming/cophi/projects/mtpnet/Text-code/NL-code-search-Adv/python/model", type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

# ## Other parameters
parser.add_argument("--train_data_file", default=None, type=str,
                    help="The input training data file (a text file).")
# parser.add_argument("--eval_data_file", default=None, type=str,
#                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
# parser.add_argument("--test_data_file", default=None, type=str,
#                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

parser.add_argument("--model_type", default="roberta", type=str,
                    help="The model architecture to be fine-tuned.")
parser.add_argument("--model_name_or_path", default=None, type=str,
                    help="The model checkpoint for weights initialization.")

# parser.add_argument("--mlm", action='store_true',
#                     help="Train with masked-language modeling loss instead of language modeling.")
# parser.add_argument("--mlm_probability", type=float, default=0.15,
#                     help="Ratio of tokens to mask for masked language modeling loss")

parser.add_argument("--config_name", default="", type=str,
                    help="Optional pretrained config name or path if not the same as model_name_or_path")
parser.add_argument("--tokenizer_name", default="/home/yiming/cophi/projects/codebert-base", type=str,
                    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
parser.add_argument("--cache_dir", default="/home/yiming/cophi/projects/codebert-base", type=str,
                    help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
parser.add_argument("--block_size", default=256, type=int,
                    help="Optional input sequence length after tokenization."
                            "The training dataset will be truncated in block of this size for training."
                            "Default to the model max input length for single sentence inputs (take into account special tokens).")
# parser.add_argument("--do_train", action='store_true',
#                     help="Whether to run training.")
# parser.add_argument("--do_eval", action='store_true',
#                     help="Whether to run eval on the dev set.")
# parser.add_argument("--do_test", action='store_true',
#                     help="Whether to run eval on the dev set.")
# parser.add_argument("--evaluate_during_training", action='store_true',
#                     help="Run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--train_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training.")
# parser.add_argument("--eval_batch_size", default=4, type=int,
#                     help="Batch size per GPU/CPU for evaluation.")
# parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                     help="Number of updates steps to accumulate before performing a backward/update pass.")
# parser.add_argument("--learning_rate", default=5e-5, type=float,
#                     help="The initial learning rate for Adam.")
# parser.add_argument("--weight_decay", default=0.0, type=float,
#                     help="Weight deay if we apply some.")
# parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                     help="Epsilon for Adam optimizer.")
# parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                     help="Max gradient norm.")
# parser.add_argument("--num_train_epochs", default=1.0, type=float,
#                     help="Total number of training epochs to perform.")
# parser.add_argument("--max_steps", default=-1, type=int,
#                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
# parser.add_argument("--warmup_steps", default=0, type=int,
#                     help="Linear warmup over warmup_steps.")

# parser.add_argument('--logging_steps', type=int, default=50,
#                     help="Log every X updates steps.")
# parser.add_argument('--save_steps', type=int, default=50,
#                     help="Save checkpoint every X updates steps.")
# parser.add_argument('--save_total_limit', type=int, default=None,
#                     help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
# parser.add_argument("--eval_all_checkpoints", action='store_true',
#                     help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
# parser.add_argument("--no_cuda", action='store_true',
#                     help="Avoid using CUDA when available")
# parser.add_argument('--overwrite_output_dir', action='store_true',
#                     help="Overwrite the content of the output directory")
# parser.add_argument('--overwrite_cache', action='store_true',
#                     help="Overwrite the cached training and evaluation sets")
# parser.add_argument('--seed', type=int, default=42,
#                     help="random seed for initialization")
# parser.add_argument('--epoch', type=int, default=42,
#                     help="random seed for initialization")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
# parser.add_argument('--fp16_opt_level', type=str, default='O1',
#                     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#                             "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
# parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
# parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

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
# MAX_EPOCH = 1
VIS_MODEL_NAME = 'base_dvi' ### saved_as VIS_MODEL_NAME.pth


# Define hyperparameters
GPU_ID = 1
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
print("device", DEVICE)

# import Model.model as subject_model
# net = eval("subject_model.{}()".format(NET))
net = "Model"

# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

# MODEL_CLASSES = {
#     'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
#     'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
# }

# import logging
# from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm, trange
# logger = logging.getLogger(__name__)
# CUDA_VISIBLE_DEVICES=2,3

# class InputFeatures(object):
#     """A single training/test features for a example."""

#     def __init__(self,
#                  code_tokens,
#                  code_ids,
#                  nl_tokens,
#                  nl_ids,
#                  url,
#                  idx,

#                  ):
#         self.code_tokens = code_tokens
#         self.code_ids = code_ids
#         self.nl_tokens = nl_tokens
#         self.nl_ids = nl_ids
#         self.url = url
#         self.idx = idx

# def convert_examples_to_features(js, tokenizer, args):
#     # code
#     if 'code_tokens' in js:
#         code = ' '.join(js['code_tokens'])
#     else:
#         code = ' '.join(js['function_tokens'])
#     code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
#     code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
#     code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
#     padding_length = args.block_size - len(code_ids)
#     code_ids += [tokenizer.pad_token_id] * padding_length

#     nl = ' '.join(js['docstring_tokens'])
#     nl_tokens = tokenizer.tokenize(nl)[:args.block_size - 2]
#     nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
#     nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
#     padding_length = args.block_size - len(nl_ids)
#     nl_ids += [tokenizer.pad_token_id] * padding_length

#     return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'], js['idx'])

# class TextDataset(Dataset):
#     def __init__(self, tokenizer, args, file_path=None):
#         self.examples = []
#         data = []
#         with open(file_path) as f:
#             for i, line in enumerate(f):
#                 # if i>200:
#                 #     break
#                 line = line.strip()
#                 js = json.loads(line)
#                 data.append(js)
#         for js in data:
#             self.examples.append(convert_examples_to_features(js, tokenizer, args))
#         if 'train' in file_path:
#             for idx, example in enumerate(self.examples[:1]):
#                 logger.info("*** Example ***")
#                 logger.info("idx: {}".format(idx))
#                 logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
#                 logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
#                 logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
#                 logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))

# config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
# config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
#                                         cache_dir=args.cache_dir if args.cache_dir else None)
# config.num_labels = 1
# tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
#                                             do_lower_case=args.do_lower_case,
#                                             cache_dir=args.cache_dir if args.cache_dir else None)
# if args.block_size <= 0:
#     args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
# args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
# if args.model_name_or_path:
#     model = model_class.from_pretrained(args.model_name_or_path,
#                                         config=config,
#                                         cache_dir=args.cache_dir if args.cache_dir else None)
# else:
#     model = model_class(config)

# from Model.model import Model
# model = Model(model, config, tokenizer, args)
# # logger.warning("model")
# train_dataset = TextDataset(tokenizer, args, args.train_data_file)
# """ Train the model """
# args.per_gpu_train_batch_size = args.train_batch_size
# # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
# train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
# logger.warning("aaaa")

# args.num_train_epochs = args.epoch

# if args.fp16:
#     try:
#         from apex import amp
#     except ImportError:
#         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

# # multi-gpu training (should be after apex fp16 initialization)
# # if args.n_gpu > 1:
# #     model = torch.nn.DataParallel(model)

# # Distributed training (should be after apex fp16 initialization)
# if args.local_rank != -1:
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                         output_device=args.local_rank,
#                                                         find_unused_parameters=True)

# # checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
# # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
# # optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
# # if os.path.exists(scheduler_last):
# #     scheduler.load_state_dict(torch.load(scheduler_last))
# # if os.path.exists(optimizer_last):
# #     optimizer.load_state_dict(torch.load(optimizer_last))

# # Train!
# logger.info("***** Running vector generation *****")
# logger.info("  Num examples = %d", len(train_dataset))
# logger.info("  Num Epochs = %d", args.num_train_epochs)
# logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)

# # for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
# for idx in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
#     # 每一轮记录checkpoint
#     output_dir = os.path.join(args.output_dir, 'Epoch_{}'.format(idx+1))
#     model_to_save = model.module if hasattr(model, 'module') else model
#     ckpt_output_path = os.path.join(output_dir, 'subject_model.pth')
#     # model.load_state_dict(torch.load(ckpt_output_path, map_location=torch.device('cpu')),strict=False) 
#     model.load_state_dict(torch.load(ckpt_output_path),strict=False) 
#     model.to(DEVICE)
#     # 每一轮记录表征
#     logger.info("Saving training feature")
#     train_dataloader_bs1 = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
#     code_feature, nl_feature = [], []
#     for batch in tqdm(train_dataloader_bs1):
#         code_inputs = batch[0].to(DEVICE)
#         nl_inputs = batch[1].to(DEVICE)
#         # code_inputs = batch[0].to(torch.device('cpu'))
#         # nl_inputs = batch[1].to(torch.device('cpu'))
#         model.eval()
#         with torch.no_grad():
#             lm_loss, code_vec, nl_vec = model(code_inputs, nl_inputs)
#             # cf, nf = model.feature(code_inputs=code_inputs, nl_inputs=nl_inputs)
#             code_feature.append(code_vec.cpu().detach().numpy())
#             nl_feature.append(nl_vec.cpu().detach().numpy())
#     code_feature = np.concatenate(code_feature, 0)
#     nl_feature = np.concatenate(nl_feature, 0)
#     print(code_feature.shape, nl_feature.shape)
#     # code_feature_output_path = os.path.join(output_dir, 'code_feature.npy')
#     # nl_feature_output_path = os.path.join(output_dir, 'nl_feature.npy')
#     # with open(code_feature_output_path, 'wb') as f1, open(nl_feature_output_path, 'wb') as f2:
#     #     np.save(code_feature, f1)
#     #     np.save(code_feature, f2)


########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
# Define data_provider
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)
PREPROCESS = args.preprocess
if PREPROCESS:
    data_provider._meta_data()
#     if B_N_EPOCHS >0:
#         data_provider._estimate_boundary(LEN // 10, l_bound=L_BOUND)

# Define visualization models
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)


class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from, probs):
        # get negative samples
        batch_size = embedding_to.shape[0]
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]
        neg_num = len(embedding_neg_from)

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)

        distance_embedding = torch.cat(
            (
                positive_distance,
                negative_distance,
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        probabilities_graph = torch.cat(
            (torch.ones(batch_size).to(self.DEVICE), torch.zeros(neg_num).to(self.DEVICE)), dim=0,
        )

        # probabilities_graph = torch.cat(
        #     (probs.to(self.DEVICE), torch.zeros(neg_num).to(self.DEVICE)), dim=0,
        # )

        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )   

        return torch.mean(ce_loss)

class DVILoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, device):
        super(DVILoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model,probs):
      
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from, probs)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss
    



umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define Projector
projector = VISProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)



start_flag = 1

prev_model = VisModel(ENCODER_DIMS, DECODER_DIMS)

for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    # Define DVI Loss
    if start_flag:
        temporal_loss_fn = DummyTemporalLoss(DEVICE)
        criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
        start_flag = 0
    else:
        # TODO AL mode, redefine train_representation
        prev_data = data_provider.train_representation(iteration-EPOCH_PERIOD)
        prev_data = prev_data.reshape(prev_data.shape[0],prev_data.shape[1])
        curr_data = data_provider.train_representation(iteration)
        curr_data = curr_data.reshape(curr_data.shape[0],curr_data.shape[1])
        print(prev_data.shape, curr_data.shape)
        t_1= time.time()
        npr = torch.tensor(find_neighbor_preserving_rate(prev_data, curr_data, N_NEIGHBORS)).to(DEVICE)
        t_2= time.time()
     
        # temporal_loss_fn = TemporalLoss(w_prev, DEVICE)
        # criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr,device=DEVICE)
        temporal_loss_fn = DummyTemporalLoss(DEVICE)
        criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    # Define Edge dataset


    
    

    t0 = time.time()
    ##### construct the spitial complex
    spatial_cons = SingleEpochTextSpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, net)
    edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()
    t1 = time.time()

    print('complex-construct:', t1-t0)

    probs = probs / (probs.max()+1e-3)
    eliminate_zeros = probs> 1e-3    #1e-3
    edge_to = edge_to[eliminate_zeros]
    edge_from = edge_from[eliminate_zeros]
    probs = probs[eliminate_zeros]
    
    labels_non_boundary = np.zeros(len(edge_to))


    # pred_list = data_provider.get_pred(iteration, feature_vectors)
    pred_list = np.zeros(feature_vectors.shape)
    dataset = VisDataHandler(edge_to, edge_from, feature_vectors, attention, probs,pred_list)

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

    trainer = BaseTextTrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

    t2=time.time()
    trainer.train(PATIENT, MAX_EPOCH, data_provider,iteration)
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

for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    train_data = data_provider.train_representation(iteration)
    train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
    emb = projector.batch_project(iteration, train_data)
    inv = projector.batch_inverse(iteration, emb)
    save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
    train_data_loc = os.path.join(save_dir, "embedding.npy")
    np.save(train_data_loc, emb)
#     inv_loc = os.path.join(save_dir, "inv.npy")
#     np.save(inv_loc, inv)
#     cluster_rep_loc = os.path.join(save_dir, "cluster_centers.npy")
#     cluster_rep = np.load(cluster_rep_loc)
#     emb = projector.batch_project(iteration, cluster_rep)
#     inv = projector.batch_inverse(iteration, emb)
#     inv_loc = os.path.join(save_dir, "inv_cluster.npy")
#     np.save(inv_loc, inv)
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
    vis.save_scale_bgimg(i)

# emb = projector.batch_project(data_provider)

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
# evaluator = Evaluator(data_provider, projector)




# Evaluation_NAME = '{}_eval'.format(VIS_MODEL_NAME)
# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
#     evaluator.save_epoch_eval(i, 15, temporal_k=5, file_name="{}".format(Evaluation_NAME))
