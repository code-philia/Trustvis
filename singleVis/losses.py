from abc import ABC, abstractmethod
import torch
from torch import nn
from singleVis.backend import convert_distance_to_probability, compute_cross_entropy
from scipy.special import softmax
import torch.nn.functional as F


        
import torch
torch.manual_seed(0)  # fixed seed
torch.cuda.manual_seed_all(0)
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

# Set the random seed for numpy

"""Losses modules for preserving four propertes"""
# https://github.com/ynjnpa/VocGAN/blob/5339ee1d46b8337205bec5e921897de30a9211a1/utils/stft_loss.py for losses module

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

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

    def forward(self, embedding_to, embedding_from, probs, pred_edge_to, pred_edge_from):
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

        batch_margin = (positive_distance_mean + negative_distance_mean) / 2

        # ##### strategy: if pred dissimilar change to negative
        pred_edge_to_Res = pred_edge_to.argmax(axis=1)
        pred_edge_from_Res = pred_edge_from.argmax(axis=1)

        # dynamic labeling
        # condition = (pred_edge_to_Res.to(self.DEVICE) != pred_edge_from_Res.to(self.DEVICE)) & (positive_distance.to(self.DEVICE) < negative_distance_mean.to(self.DEVICE))
        # probs[condition] = 0

        is_pred_same = (pred_edge_to_Res.to(self.DEVICE) == pred_edge_from_Res.to(self.DEVICE))
        dynamic_margin = (1.0 - is_pred_same.float()) * batch_margin
        # 计算 margin loss
        margin_loss = F.relu(dynamic_margin.to(self.DEVICE) - positive_distance.to(self.DEVICE)).mean()


    
        # adjusted_positive_distance = positive_distance + torch.abs(dynamic_margin)
        # adjusted_positive_distance = positive_distance - torch.abs(dynamic_margin * 0.5)
        # adjusted_positive_distance = torch.clamp(adjusted_positive_distance, min=0)


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

        return torch.mean(ce_loss) + margin_loss


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

    def forward(self, edge_to, edge_from, a_to, a_from, curr_model,probs,pred_edge_to, pred_edge_from):
        outputs = curr_model( edge_to, edge_from)
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        # TODO stop gradient edge_to_ng = edge_to.detach().clone()

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from).to(self.device)
        umap_l = self.umap_loss(embedding_to, embedding_from, probs,pred_edge_to, pred_edge_from).to(self.device)
        temporal_l = self.temporal_loss(curr_model).to(self.device)

        loss = umap_l + self.lambd1 * recon_l + self.lambd2 * temporal_l

        return umap_l, self.lambd1 *recon_l, self.lambd2 *temporal_l, loss



    
# TODO delete after eval
class SementicUmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, data_provider, iteration, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(SementicUmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.iteration = iteration

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from,recon_to, recon_from, probs,pred_org_to, pred_org_from, margin=0):
        batch_size = embedding_to.shape[0]

        max_pred_org_to = torch.argmax(torch.Tensor(pred_org_to), dim=1)
        max_pred_org_from = torch.argmax(torch.Tensor(pred_org_from), dim=1)
 
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]


        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)
        probs = self.reweight_samples(max_pred_org_to, max_pred_org_from, probs)
        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (probs, torch.zeros(batch_size * self._negative_sample_rate).to(self.DEVICE)), dim=0,
        )
        # probabilities_graph = torch.cat(
        #     (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        # )
        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )


        return torch.mean(ce_loss) 
    

    def reweight_samples(self, max_pred_org_to, max_pred_org_from, probs, alpha=0.1, beta=0.8):
        """

        base on init probs and prediction adjust sample's weight

        parameters:
        pred_recon_to: Tensor, reconstriction prediction to
        pred_recon_from: Tensor, reconstriction prediction from
        pred_org_to: Tensor, original prediction to
        pred_org_from: Tensor, original prediction from
        probs: Tensor, original hight-dimensional weights
        alpha: float, for weight adjust
        beta: float, for weight adjust

        return:
        Tensor, adjusted weight
        """
        # pred_recon_to = torch.Tensor(pred_recon_to).to(self.DEVICE)
        # pred_recon_from = torch.Tensor(pred_recon_from).to(self.DEVICE)
        # pred_org_to = torch.Tensor(pred_org_to).to(self.DEVICE)
        # pred_org_from = torch.Tensor(pred_org_from).to(self.DEVICE)
        # probs = probs.to(self.DEVICE)

        # max_pred_recon_to = torch.argmax(pred_recon_to, dim=1)
        # max_pred_recon_from = torch.argmax(pred_recon_from, dim=1)
        # max_pred_org_to = torch.argmax(pred_org_to, dim=1)
        # max_pred_org_from = torch.argmax(pred_org_from, dim=1)

        # max_pred_recon_to = max_pred_recon_to[matching_indices]
        # max_pred_recon_from = max_pred_recon_from[matching_indices]
        # max_pred_org_to = max_pred_org_to[matching_indices]
        # max_pred_org_from = max_pred_org_from[matching_indices]
        # probs = probs[matching_indices]


        # increase_weight_mask = (max_pred_recon_to == max_pred_org_to) & (max_pred_recon_from == max_pred_org_from)
        # decrease_weight_mask = ~(increase_weight_mask)
    
        # probs[increase_weight_mask] = probs[increase_weight_mask] * beta + (1 - beta) * alpha
        # probs[decrease_weight_mask] = probs[decrease_weight_mask] * (1 - alpha)
        
        # 如果最大预测值下标相同，则将对应的 probs 设置为 1
        same_max_pred_org_mask = (max_pred_org_to == max_pred_org_from)
        # probs[same_max_pred_org_mask] = 1.0
        # 最大预测值下标不同，但重建预测相同，施加斥力使 probs 减小
        diff_max_pred_org_same_max_pred_recon_mask = ~same_max_pred_org_mask
        probs[diff_max_pred_org_same_max_pred_recon_mask] = 0


        # 其他情况，根据 alpha 调整 probs
        # probs[~same_max_pred_mask] *= alpha

        # # 计算 pred_org_to 和 pred_org_from 的相似度
        # similarity = (pred_org_to == pred_org_from).float()

        # # 计算 pred_org_to 和 pred_org_from 在每个维度上的相似度
        # similarity_per_dim = (pred_org_to == pred_org_from).float()  # shape: [N, 10]

        # # 将相似度聚合到一个维度
        # similarity = torch.mean(similarity_per_dim, dim=1)  # shape: [N]


        # # 调整相似情况下的权重
        # adjusted_probs = probs * (1 - similarity) + similarity * alpha

        # # 考虑重构预测与原始预测的不同
        # recon_diff_to = torch.abs(pred_recon_to - pred_org_to)
        # recon_diff_from = torch.abs(pred_recon_from - pred_org_from)
        # adjusted_probs *= (1 - beta * (recon_diff_to + recon_diff_from))

        return probs
    
    
# TODO delete after eval
class LogitUmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, data_provider, iteration, _a=1.0, _b=1.0, repulsion_strength=1.0):
        super(LogitUmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = _a,
        self._b = _b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)
        self.data_provider = data_provider
        self.iteration = iteration

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from, org_to, org_from, recon_to, recon_from, probs, margin=0):
        batch_size = embedding_to.shape[0]
        combined_samples = torch.cat((org_to, org_from, recon_to, recon_from), dim=0)
        combined_probs = self.data_provider.get_pred(self.iteration, combined_samples.cpu().detach().numpy(),0)
        org_to_p, org_from_p, recon_to_p, recon_from_p = np.split(combined_probs, 4, axis=0)
        max_indices_to_p = np.argmax(org_to_p, axis=1)
        max_indices_from_p = np.argmax(org_from_p, axis=1)
        recon_max_to_p = np.array([recon_to_p[i, idx] for i, idx in enumerate(max_indices_to_p)])
        recon_max_from_p = np.array([recon_from_p[i, idx] for i, idx in enumerate(max_indices_from_p)])

        # 计算这些对应下标的值的绝对值差异
        diff_to_p = np.abs(org_to_p[np.arange(len(org_to_p)), max_indices_to_p] - recon_max_to_p)
        diff_from_p = np.abs(org_from_p[np.arange(len(org_from_p)), max_indices_from_p] - recon_max_from_p)

        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]

        positive_distance = torch.norm(embedding_to - embedding_from, dim=1)
        negative_distance = torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        distance_embedding = torch.cat((positive_distance, negative_distance), dim=0)

        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        # set true probabilities based on negative sampling
        num_neg_samples = embedding_neg_to.shape[0]  # valied negative samples
        diff_to_p_tensor = torch.from_numpy(diff_to_p).float().to(probs.device)
        diff_from_p_tensor = torch.from_numpy(diff_from_p).float().to(probs.device)

        # 确保这些张量与 probs 的形状匹配
        average_diff_tensor = (diff_to_p_tensor + diff_from_p_tensor) / 2

        # 计算加权的概率
        #TODO probs
        weighted_probs = torch.ones(batch_size).to(device=self.DEVICE) * average_diff_tensor
        probabilities_graph = torch.cat(
            (weighted_probs, torch.zeros(num_neg_samples).to(device=self.DEVICE)), dim=0,
         )
        

   
        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )

        # margin_loss = F.relu(margin - positive_distance).mean()

        # total_loss = torch.mean(ce_loss)  + margin_loss

        return ce_loss
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

    def forward(self, edge_to, edge_from, a_to, a_from, outputs):
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]

        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from, a_to, a_from)
        # recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        umap_l = self.umap_loss(embedding_to, embedding_from)

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


