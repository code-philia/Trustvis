from abc import ABC, abstractmethod
import torch
from torch import nn
from singleVis.backend import convert_distance_to_probability, compute_cross_entropy
from scipy.special import softmax
import torch.nn.functional as F
import torch.optim as optim
import os


        
import torch
torch.manual_seed(0)  # fixed seed
torch.cuda.manual_seed_all(0)
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

import json
from datetime import datetime
# Set the random seed for numpy

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py for losses module

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class MyModel(nn.Module):
    def __init__(self, initial_tensor):
        super(MyModel, self).__init__()
        self.learnable_matrix = nn.Parameter(initial_tensor.clone().detach())

class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device,  data_provider, epoch, net, fixed_number = 5, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.epoch = epoch
        self.net = net
        self.model_path = os.path.join(self.data_provider.content_path, "Model")
        self.fixed_number = fixed_number

        model_location = os.path.join(self.model_path, "{}_{:d}".format('Epoch', epoch), "subject_model.pth")
        self.net.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")),strict=False)
        self.net.to(self.DEVICE)
        self.net.train()

        for param in net.parameters():
            param.requires_grad = False

        self.pred_fn = self.net.prediction

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, edge_to_idx, edge_from_idx,embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from,curr_model,iteration):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        pred_edge_to_neg_Res = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        pred_repeat_neg = torch.repeat_interleave(pred_edge_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]
        pred_edge_from_neg_Res = pred_repeat_neg[randperm]
        indicates = self.filter_neg(pred_edge_from_neg_Res, pred_edge_to_neg_Res)

        #### strategy confidence: filter negative
        embedding_neg_to = embedding_neg_to[indicates]
        embedding_neg_from = embedding_neg_from[indicates]

        neg_num = len(embedding_neg_from)

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        #  distances between samples (and negative samples)
        positive_distance_mean = torch.mean(positive_distance)
        negative_distance_mean = torch.mean(negative_distance)

        #### dynamic labeling
        pred_edge_to_Res = pred_edge_to.argmax(axis=1)
        pred_edge_from_Res = pred_edge_from.argmax(axis=1)

        is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
        is_pred_same = is_pred_same.to(self.DEVICE)
        pred_edge_to = pred_edge_to.to(self.DEVICE)
        pred_edge_from = pred_edge_from.to(self.DEVICE)

        recon_pred_to_Res = recon_pred_edge_to.argmax(axis=1)
        recon_pred_from_Res = recon_pred_edge_from.argmax(axis=1)


        temp = 0.001
        recon_pred_to_softmax = F.softmax(recon_pred_edge_to / temp, dim=-1)
        recon_pred_from_softmax = F.softmax(recon_pred_edge_from / temp, dim=-1)

        pred_to_softmax = F.softmax(pred_edge_to / temp, dim=-1)
        pred_from_softmax = F.softmax(pred_edge_from / temp, dim=-1)
        
        recon_pred_to_softmax = torch.Tensor(recon_pred_to_softmax.to(self.DEVICE))
        recon_pred_from_softmax = torch.Tensor(recon_pred_from_softmax.to(self.DEVICE))
        #### umap loss
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
            (probs, torch.zeros(neg_num).to(self.DEVICE)), dim=0,
        )

        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )  


        batch_margin = positive_distance_mean +  (negative_distance_mean - positive_distance_mean) * (1-probs)
        init_margin = (1.0 - is_pred_same.float()) * batch_margin

        if iteration > self.fixed_number:
            margin = self.newton_step_with_regularization(edge_to_idx, edge_from_idx,init_margin, is_pred_same, 
                                                      edge_to[~is_pred_same],edge_from[~is_pred_same], probs[~is_pred_same],
                                                      embedding_to[~is_pred_same],embedding_from[~is_pred_same],curr_model,
                                                      pred_to_softmax[~is_pred_same], pred_from_softmax[~is_pred_same],positive_distance_mean,negative_distance_mean)
        else:
            margin = init_margin
        
        # print(margin[~is_pred_same].mean().item(),positive_distance.mean().item(), positive_distance[~is_pred_same].mean().item())
        
        # print("dynamic marin", margin[~is_pred_same].mean())
        # print("margin", margin.mean().item())
        margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()
        
        umap_l = torch.mean(ce_loss).to(self.DEVICE) 
        margin_loss = margin_loss.to(self.DEVICE)

        if torch.isnan(margin_loss):
            margin_loss = torch.tensor(0.0).to(margin_loss.device)

        return umap_l, margin_loss, umap_l+margin_loss

    def filter_neg(self, neg_pred_from, neg_pred_to, delta=1e-1):
        neg_pred_from = neg_pred_from.cpu().detach().numpy()
        neg_pred_to = neg_pred_to.cpu().detach().numpy()
        neg_conf_from =  np.amax(softmax(neg_pred_from, axis=1), axis=1)
        neg_conf_to =  np.amax(softmax(neg_pred_to, axis=1), axis=1)
        neg_pred_edge_from_Res = neg_pred_from.argmax(axis=1)
        neg_pred_edge_to_Res = neg_pred_to.argmax(axis=1)
        condition1 = (neg_pred_edge_from_Res==neg_pred_edge_to_Res)
        condition2 = (neg_conf_from==neg_conf_to)
        # condition2 = (np.abs(neg_conf_from - neg_conf_to)< delta)
        indices = np.where(~(condition1 & condition2))[0]
        return indices
    
    def newton_step_with_regularization(self, edge_to_idx, edge_from_idx, dynamic_margin, is_pred_same, edge_to, edge_from, probs, emb_to, emb_from, curr_model, pred_to_softmax, pred_from_softmax, positive_distance_mean, negative_distance_mean, epsilon=1e-4):
        # Ensure the input tensors require gradient
        for tensor in [edge_to, edge_from, emb_to, emb_from]:
            tensor.requires_grad_(True)

        # umap loss
    
        distance_embedding = torch.norm(emb_to - emb_from, dim=1)
        probabilities_distance = convert_distance_to_probability(distance_embedding, self.a, self.b)
        probabilities_graph = probs
        _, _, ce_loss = compute_cross_entropy(probabilities_graph, probabilities_distance, repulsion_strength=self._repulsion_strength)

        # Create a tensor of ones with the same size as ce_loss
        ones = torch.ones_like(ce_loss)

        # Compute gradient 
        grad = torch.autograd.grad(ce_loss, emb_to, grad_outputs=ones, create_graph=True)[0]
        # Compute gradient for emb_from
        grad_emb_from = torch.autograd.grad(ce_loss, emb_from, grad_outputs=ones, create_graph=True)[0] 

        ################################################################################## analysis grad end  ############################################################################################

        # use learning rate approximate the y_next
        next_emb_to = emb_to - 1 * grad
        next_emb_from = emb_from - 1 * grad_emb_from

        """
        strategy 1: gen y* from yi, and then push y* to yj then calculate || y*-yi || 
        strategy 2: gen y* from yj, and then pull y* to yi then calculate || y*-yi || 
        strategy 3: gen yi* and yj * from yj an yi, and push yi* to yj di = || yi* - yi||,and push yi* to yi, dj = || yj* - yj||, margin = max(di,dj)
        """
        """ strategy 1 """
        # metrix = torch.tensor(next_emb_to, dtype=torch.float, requires_grad=True)
        # strategy 2
        # metrix = torch.tensor(next_emb_from, dtype=torch.float, requires_grad=True)
        """ strategy 3 """
        metrix = torch.tensor(torch.cat((next_emb_to, next_emb_from),dim=0), dtype=torch.float, requires_grad=True)

        for param in curr_model.parameters():
            param.requires_grad = False


        # loss = pred_from_softmax - torch.mean(torch.pow(pred_from_softmax - F.softmax(inv / 0.01, dim=-1), 2),1)
        optimizer = optim.Adam([metrix], lr=0.01)
        # 训练循环
        for epoch in range(20):
            optimizer.zero_grad() 
            inv = curr_model.decoder(metrix) 
            inv_pred = self.pred_fn(inv)
            # inv_pred = torch.tensor(inv_pred, dtype=torch.float, device=self.DEVICE, requires_grad=True)
            # # 计算损失
            
            """ strategy 1 """
            # inv_pred_to_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            # loss = 10 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2))
            # strategy 2   
            
                # Calculate the three terms separately
                # first_term = torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2))
                # third_term = torch.mean(torch.pow(emb_to - metrix, 2))
                # threshold = 0.01  # 

                # if first_term.item() < threshold:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * third_term
                # else:
                #     loss = first_term + torch.mean(torch.pow(inv - edge_from, 2))
                                             # 
                # loss = 100 * torch.mean(torch.pow(pred_from_softmax - inv_pred_to_softmax, 2)) + torch.mean(torch.pow(inv - edge_from, 2)) + 0.1 * torch.mean(torch.pow(emb_to - metrix, 2))
            """ strategy 3 """
            inv_pred_softmax = F.softmax(inv_pred / 0.001, dim=-1)
            loss = 10 * torch.mean(torch.pow(torch.cat((pred_from_softmax,pred_to_softmax),dim=0) - inv_pred_softmax, 2)) + torch.mean(torch.pow(inv - torch.cat((edge_from, edge_to),dim=0), 2))

            # bp
            loss.backward(retain_graph=True)         
            optimizer.step()

            # if loss.item() < 0.1:
            #     print(f"Stopping early at epoch {epoch} as loss dropped below 0.1")
            #     break
            # if epoch % 500 == 0:
                # print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        """strategy 1 or 2"""

        # final_margin = torch.norm(emb_to - metrix, dim=1)

        """strategy 3 start"""
        margin = torch.norm( torch.cat((emb_to, emb_from),dim=0) - metrix, dim=1)
        total_length = margin.size(0)
        half_length = total_length // 2
        margin_to = margin[:half_length]
        margin_from = margin[half_length:]    
        final_margin =  torch.max(margin_to, margin_from)
        """strategy 3 end """

        # final_margin = torch.where(final_margin < positive_distance_mean.item(), dynamic_margin[~is_pred_same], final_margin)
        
        
        # margin = torch.where(margin > negative_distance_mean.item(), dynamic_margin[~is_pred_same], margin)

        final_margin = torch.max(final_margin, dynamic_margin[~is_pred_same])

        for param in curr_model.parameters():
            param.requires_grad = True

        dynamic_margin[~is_pred_same] = final_margin.to(self.DEVICE)



        return dynamic_margin

class UmapLoss_refine_conf(nn.Module):
    def __init__(self, negative_sample_rate, device,  data_provider, epoch, net, error_conf, neg_grid, pos_grid,distance_list,fixed_number = 5, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(UmapLoss_refine_conf, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.epoch = epoch
        self.net = net
        self.model_path = os.path.join(self.data_provider.content_path, "Model")
        self.fixed_number = fixed_number
        
        self.error_conf = torch.tensor(error_conf)
        self.neg_grid = torch.tensor(neg_grid)
        self.pos_grid = torch.tensor(pos_grid)
        self.distance_list = distance_list

        model_location = os.path.join(self.model_path, "{}_{:d}".format('Epoch', epoch), "subject_model.pth")
        self.net.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")),strict=False)
        self.net.to(self.DEVICE)
        self.net.train()

        for param in net.parameters():
            param.requires_grad = False

        self.pred_fn = self.net.prediction

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, edge_to_idx, edge_from_idx, embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from,curr_model,iteration):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        pred_edge_to_neg_Res = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        pred_repeat_neg = torch.repeat_interleave(pred_edge_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]
        pred_edge_from_neg_Res = pred_repeat_neg[randperm]
        indicates = self.filter_neg(pred_edge_from_neg_Res, pred_edge_to_neg_Res)

        #### strategy confidence: filter negative
        embedding_neg_to = embedding_neg_to[indicates]
        embedding_neg_from = embedding_neg_from[indicates]
        

        neg_num = len(embedding_neg_from)
        
        #### identify if the conf error is in the current pair #####################################
        conf_e_neg_from = []
        conf_e_pos_from = []
        conf_e_to_pos = []
        conf_e_to_neg = []

        ####### label the element that in the error conf
        mask_to = torch.isin(edge_to_idx, self.error_conf)
        mask_from = torch.isin(edge_from_idx, self.error_conf)
        ######## get the indicates
        conf_e_indices_to = torch.nonzero(mask_to, as_tuple=True)[0]
        conf_e_indices_from = torch.nonzero(mask_from, as_tuple=True)[0]

        filtered_to_idx = edge_to_idx[conf_e_indices_to]
        filtered_from_idx = edge_from_idx[conf_e_indices_from]

        for i,org_index in enumerate(filtered_to_idx):
            ##### get the position in the error_conf
            indicates = (self.error_conf == org_index).nonzero(as_tuple=True)[0]
            indicates= indicates[0].item()
            neg_grids = self.neg_grid[indicates].squeeze()
            # pos_grids = self.pos_grid[indicates].squeeze()
            pos_grids = self.pos_grid[indicates].unsqueeze(0) if self.pos_grid[indicates].ndim == 1 else self.pos_grid[indicates]
            
            # Append the matched negative grids and the corresponding embedding
            # org_emb = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
            conf_e_from.extend([embedding_to[i]] * len(neg_grids)) 
            conf_e_to_neg.extend(neg_grids)
            
            if self.distance_list[indicates] < 0.5:
                # print("indicates to",indicates)
                conf_e_pos_from.extend([embedding_from[i]] * len(pos_grids)) 
                conf_e_to_pos.extend(pos_grids)
        
        for i,org_index in enumerate(filtered_from_idx):
            indicates = (self.error_conf == org_index).nonzero(as_tuple=True)[0]
            indicates= indicates[0].item()
            neg_grids = self.neg_grid[indicates].squeeze()
            # pos_grids = self.pos_grid[indicates].squeeze()
            pos_grids = self.pos_grid[indicates].unsqueeze(0) if self.pos_grid[indicates].ndim == 1 else self.pos_grid[indicates]
            
            # Append the matched negative grids and the corresponding embedding
            # org_emb = torch.repeat_interleave(pred_edge_to, self._negative_sample_rate, dim=0)
            conf_e_neg_from.extend([embedding_from[i]] * len(neg_grids)) 
            conf_e_to_neg.extend(neg_grids)
            if self.distance_list[indicates] < 1:
                # print("indicates from",indicates)
                conf_e_pos_from.extend([embedding_from[i]] * len(pos_grids)) 
                conf_e_to_pos.extend(pos_grids)
        
        # Convert lists to tensors
        if len(conf_e_to_pos) > 0:
            conf_e_neg_from = torch.stack(conf_e_neg_from).to(self.DEVICE)
            conf_e_pos_from = torch.stack(conf_e_pos_from).to(self.DEVICE)
            conf_e_to_pos = torch.stack(conf_e_to_pos).to(self.DEVICE)
            conf_e_to_neg = torch.stack(conf_e_to_neg).to(self.DEVICE)
            # print("conf_e_pos_from",conf_e_pos_from.shape,"conf_e_to_pos",conf_e_to_pos.shape,"conf_e_from - conf_e_to_neg",conf_e_from.shape,conf_e_to_neg.shape)
            conf_pos_distance =  torch.norm(conf_e_pos_from - conf_e_to_pos, dim=1).to(self.DEVICE)
            conf_neg_distance = torch.norm(conf_e_neg_from - conf_e_to_neg, dim=1).to(self.DEVICE)
        else:
            conf_pos_distance = torch.tensor([], device=self.DEVICE)
            conf_neg_distance = torch.tensor([], device=self.DEVICE)
            
 

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        #  distances between samples (and negative samples)
        positive_distance_mean = torch.mean(positive_distance)
        negative_distance_mean = torch.mean(negative_distance)

        #### dynamic labeling
        pred_edge_to_Res = pred_edge_to.argmax(axis=1)
        pred_edge_from_Res = pred_edge_from.argmax(axis=1)

        is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
        is_pred_same = is_pred_same.to(self.DEVICE)
        pred_edge_to = pred_edge_to.to(self.DEVICE)
        pred_edge_from = pred_edge_from.to(self.DEVICE)

        recon_pred_to_Res = recon_pred_edge_to.argmax(axis=1)
        recon_pred_from_Res = recon_pred_edge_from.argmax(axis=1)


        temp = 0.001
        recon_pred_to_softmax = F.softmax(recon_pred_edge_to / temp, dim=-1)
        recon_pred_from_softmax = F.softmax(recon_pred_edge_from / temp, dim=-1)

        pred_to_softmax = F.softmax(pred_edge_to / temp, dim=-1)
        pred_from_softmax = F.softmax(pred_edge_from / temp, dim=-1)
        
        recon_pred_to_softmax = torch.Tensor(recon_pred_to_softmax.to(self.DEVICE))
        recon_pred_from_softmax = torch.Tensor(recon_pred_from_softmax.to(self.DEVICE))
        #### umap loss
        distance_embedding = torch.cat(
            (
                positive_distance,
                conf_pos_distance,
                negative_distance,
                conf_neg_distance
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        probabilities_graph = torch.cat(
        (
            probs,
            torch.ones(len(conf_pos_distance)).to(self.DEVICE), # ground truth grid points
            # torch.zeros(len(conf_neg_distance)).to(self.DEVICE)
            torch.zeros(neg_num + len(conf_neg_distance)).to(self.DEVICE)
            # torch.zeros(neg_num + len(conf_neg_distance)).to(self.DEVICE)
        ),dim=0)

        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )  


        batch_margin = positive_distance_mean +  (negative_distance_mean - positive_distance_mean) * (1-probs)
        init_margin = (1.0 - is_pred_same.float()) * batch_margin

        margin = init_margin
        
        margin_loss = F.relu(margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()
        
        umap_l = torch.mean(ce_loss).to(self.DEVICE) 
        if torch.isnan(umap_l):
            umap_l = torch.tensor(0.0).to(DEVICE)
        margin_loss = margin_loss.to(self.DEVICE)

        if torch.isnan(margin_loss):
            margin_loss = torch.tensor(0.0).to(margin_loss.device)

        return umap_l, margin_loss, umap_l+margin_loss

    def filter_neg(self, neg_pred_from, neg_pred_to, delta=1e-1):
        neg_pred_from = neg_pred_from.cpu().detach().numpy()
        neg_pred_to = neg_pred_to.cpu().detach().numpy()
        neg_conf_from =  np.amax(softmax(neg_pred_from, axis=1), axis=1)
        neg_conf_to =  np.amax(softmax(neg_pred_to, axis=1), axis=1)
        neg_pred_edge_from_Res = neg_pred_from.argmax(axis=1)
        neg_pred_edge_to_Res = neg_pred_to.argmax(axis=1)
        condition1 = (neg_pred_edge_from_Res==neg_pred_edge_to_Res)
        condition2 = (neg_conf_from==neg_conf_to)
        # condition2 = (np.abs(neg_conf_from - neg_conf_to)< delta)
        indices = np.where(~(condition1 & condition2))[0]
        return indices   

class DVILoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, device):
        super(DVILoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to_idx, edge_from_idx, edge_to, edge_from, a_to, a_from, curr_model,probs,pred_edge_to, pred_edge_from,recon_pred_edge_to,recon_pred_edge_from,iteration):
      
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]


        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l,new_l,total_l = self.umap_loss(edge_to_idx, edge_from_idx, embedding_to, embedding_from, probs,pred_edge_to, pred_edge_from,edge_to, edge_from,recon_to, recon_from,a_to, a_from,recon_pred_edge_to,recon_pred_edge_from, curr_model,iteration)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = total_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, new_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss
    


class ReconstructionLoss(nn.Module):
    def __init__(self, beta=1.0,alpha=0.5,scale_factor=0.1):
        super(ReconstructionLoss, self).__init__()
        self._beta = beta
        self._alpha = alpha
        self.scale_factor = scale_factor

    def forward(self, edge_to, edge_from, recon_to, recon_from, a_to, a_from):
        loss1 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_to), self._beta), torch.pow(edge_to - recon_to, 2)), 1))
        loss2 = torch.mean(torch.mean(torch.multiply(torch.pow((1+a_from), self._beta), torch.pow(edge_from - recon_from, 2)), 1))
        return (loss1 + loss2) /2


# TODO delete
class BoundaryAwareLoss(nn.Module):
    def __init__(self, umap_loss, device, scale_factor=0.1,margin=3):
        super(BoundaryAwareLoss, self).__init__()
        self.umap_loss = umap_loss
        self.device = device
        self.scale_factor = scale_factor
        self.margin = margin
    
    def forward(self, edge_to, edge_from, model,probs):
        outputs = model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]     
        recon_to, recon_from = outputs["recon"]
        
        # reconstruction loss - recon_to, recon_from close to edge_to, edge_from
        reconstruction_loss_to = F.mse_loss(recon_to, edge_to)
        reconstruction_loss_from = F.mse_loss(recon_from, edge_from)
        recon_loss = reconstruction_loss_to + reconstruction_loss_from


        umap_l = self.umap_loss(embedding_to, embedding_from, edge_to, edge_from, recon_to, recon_from, probs, self.margin).to(self.device)
 
        # return self.scale_factor * umap_l +  0.2 * recon_loss
        return 0.1 * umap_l + 0.1* recon_loss
class BoundaryDistanceConsistencyLoss(nn.Module):
    def __init__(self, data_provider, iteration, device):
        super(BoundaryDistanceConsistencyLoss, self).__init__()
        self.data_provider = data_provider
        self.iteration = iteration
        self.device = device
    def forward(self, samples, recon_samples):
        combined_samples = torch.cat((samples, recon_samples), dim=0)
        combined_probs = self.data_provider.get_pred(self.iteration, combined_samples.cpu().detach().numpy(),0)
   
        original_probs, recon_probs = np.split(combined_probs, 2, axis=0)

        original_boundary_distances = self.calculate_boundary_distances(original_probs)
        recon_boundary_distances = self.calculate_boundary_distances(recon_probs)
        
        correlation, _ = spearmanr(original_boundary_distances, recon_boundary_distances)
        consistency_loss = 1 - abs(correlation)

        return torch.tensor(consistency_loss, requires_grad=True)

    def calculate_boundary_distances(self, probs):
        top_two_probs = np.sort(probs, axis=1)[:, -2:]
        return top_two_probs[:, 1] - top_two_probs[:, 0]

        
class TrustvisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, bon_con_loss, lambd1, lambd2, device):
        super(TrustvisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.bon_con_loss = bon_con_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model):
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)
        bon_con_loss = self.bon_con_loss(torch.cat((edge_to,edge_from), dim=0), torch.cat((recon_to,recon_from), dim=0))

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l + bon_con_loss

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, bon_con_loss, loss


class SmoothnessLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(SmoothnessLoss, self).__init__()
        self._margin = margin

    def forward(self, embedding, target, Coefficient):
        loss = torch.mean(Coefficient * torch.clamp(torch.norm(embedding-target, dim=1)-self._margin, min=0))
        return loss


class SingleVisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, lambd):
        super(SingleVisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.lambd = lambd

    def forward(self, edge_to, edge_from, a_to, a_from, outputs, probs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        # recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        umap_l = self.umap_loss(embedding_to, embedding_from, probs)

        loss = umap_l + self.lambd * recon_l

        return umap_l, recon_l, loss

class HybridLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, smooth_loss, lambd1, lambd2):
        super(HybridLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.smooth_loss = smooth_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2

    def forward(self, edge_to, edge_from, a_to, a_from, embeded_to, coeff, outputs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        umap_l = self.umap_loss(embedding_to, embedding_from)
        smooth_l = self.smooth_loss(embedding_to, embeded_to, coeff)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * smooth_l

        return umap_l, recon_l, smooth_l, loss


class TemporalLoss(nn.Module):
    def __init__(self, prev_w, device) -> None:
        super(TemporalLoss, self).__init__()
        self.prev_w = prev_w
        self.device = device
        for param_name in self.prev_w.keys():
            self.prev_w[param_name] = self.prev_w[param_name].to(device=self.device, dtype=torch.float32)

    def forward(self, curr_module):
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        # c = 0
        for name, curr_param in curr_module.named_parameters():
            # c = c + 1
            prev_param = self.prev_w[name]
            # tf dvi: diff = tf.reduce_sum(tf.math.square(w_current[j] - w_prev[j]))
            loss = loss + torch.sum(torch.square(curr_param-prev_param))
            # loss = loss + torch.norm(curr_param-prev_param, 2)
        # in dvi paper, they dont have this normalization (optional)
        # loss = loss/c
        return loss


class DummyTemporalLoss(nn.Module):
    def __init__(self, device) -> None:
        super(DummyTemporalLoss, self).__init__()
        self.device = device

    def forward(self, curr_module):
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        return loss
    

class PositionRecoverLoss(nn.Module):
    def __init__(self, device) -> None:
        super(PositionRecoverLoss, self).__init__()
        self.device = device
    def forward(self, position, recover_position):
        mse_loss = nn.MSELoss().to(self.device)
        loss = mse_loss(position, recover_position)
        return loss



class TrustALLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, device):
        super(TrustALLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.device = device

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model, outputs, edge_to_pred, edge_from_pred):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from,edge_to_pred, edge_from_pred).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss
        
       

class DVIALLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss, lambd1, lambd2, lambd3, device):
        super(DVIALLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.device = device

        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()


    def forward(self, edge_to, edge_from, a_to, a_from, curr_model, outputs,data):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()
        # if self.lambd3 != 0:
        #     data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        #     recon_data = curr_model(data,data)['recon'][0]
        #     pred_loss = self.mse_loss(data, recon_data)
        # else:
        #     pred_loss = torch.Tensor(0)
            
        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        if self.lambd3 != 0:
            data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
            recon_data = curr_model(data,data)['recon'][0]
            pred_loss = self.mse_loss(data, recon_data)
            loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l + self.lambd3 * pred_loss
            return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss, pred_loss
        else:
            loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l 
            pred_loss = torch.tensor(0.0).to(self.device)

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss, pred_loss 



class ActiveLearningLoss(nn.Module):
    def __init__(self, data_provider, iteration, device):
        super(ActiveLearningLoss, self).__init__()
        self.data_provider = data_provider
        self.iteration = iteration
        self.device = device

    def forward(self, curr_model,data):
         
        self.data = torch.tensor(data).to(device=self.device, dtype=torch.float32)
        recon_data = curr_model(self.data,self.data)['recon'][0]
        loss = self.cross_entropy_loss(self.data, recon_data)
        # normalized_loss = torch.sigmoid(loss)
        return  loss


