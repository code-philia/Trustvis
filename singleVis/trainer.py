from abc import ABC, abstractmethod
import os
import time
import gc 
import json
from tqdm import tqdm
import torch

from singleVis.visualizer import visualizer
from singleVis.eval.evaluator import Evaluator
import sys
sys.path.append('..')
from singleVis.projector import PROCESSProjector
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(0)  # fixed seed
torch.cuda.manual_seed_all(0)
from singleVis.utils import get_confidence_error_pairs
"""
1. construct a spatio-temporal complex
2. construct an edge-dataset
3. train the network

Trainer should contains
1. train_step function
2. early stop
3. ...
"""

class TrainerAbstractClass(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def reset_optim(self):
        pass

    @abstractmethod
    def update_edge_loader(self):
        pass

    @abstractmethod
    def update_vis_model(self):
        pass

    @abstractmethod
    def update_optimizer(self):
        pass

    @abstractmethod
    def update_lr_scheduler(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def train(self):
       pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def record_time(self):
        pass


class SingleVisTrainer(TrainerAbstractClass):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.DEVICE = DEVICE
        self.edge_loader = edge_loader
        self._loss = 100.0

    @property
    def loss(self):
        return self._loss

    def reset_optim(self, optim, lr_s):
        self.optimizer = optim
        self.lr_scheduler = lr_s
        print("Successfully reset optimizer!")
    
    def update_edge_loader(self, edge_loader):
        del self.edge_loader
        gc.collect()
        self.edge_loader = edge_loader
    
    def update_vis_model(self, model):
        self.model.load_state_dict(model.state_dict())
    
    def update_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def update_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def train_step(self):
        self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))

        # for data in self.edge_loader:
        for data in t:
            edge_to, edge_from, a_to, a_from, probs = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            probs = probs.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            umap_l, recon_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, outputs,probs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('umap:{:.4f}\trecon_l:{:.4f}\tloss:{:.4f}'.format(sum(umap_losses) / len(umap_losses),
                                                                sum(recon_losses) / len(recon_losses),
                                                                sum(all_loss) / len(all_loss)))
        return self.loss

    def train(self, PATIENT, MAX_EPOCH_NUMS):
        patient = PATIENT
        time_start = time.time()
        for epoch in range(MAX_EPOCH_NUMS):
            print("====================\nepoch:{}\n===================".format(epoch+1))
            prev_loss = self.loss
            loss = self.train_step()
            self.lr_scheduler.step()
            # early stop, check whether converge or not
            if prev_loss - loss < 5E-3:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

        time_end = time.time()
        time_spend = time_end - time_start
        print("Time spend: {:.2f} for training vis model...".format(time_spend))

    def load(self, file_path):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = torch.load(file_path, map_location="cpu")
        self._loss = save_model["loss"]
        self.model.load_state_dict(save_model["state_dict"])
        self.model.to(self.DEVICE)
        print("Successfully load visualization model...")

    def save(self, save_dir, file_name):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = {
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}
        save_path = os.path.join(save_dir, file_name + '.pth')
        torch.save(save_model, save_path)
        print("Successfully save visualization model...")
    
    def record_time(self, save_dir, file_name, key, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        evaluation[key] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)

class BaseTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader,DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
    
    
    def train_step(self,data_provider,iteration,epoch):
        
        projector = PROCESSProjector(self.model, data_provider.content_path, '', self.DEVICE)
        evaluator = Evaluator(data_provider, projector)
        evaluator.eval_inv_train(iteration)
        evaluator.eval_inv_test(iteration)

        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        temporal_losses = []
        new_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))


        for data in t:
            _,_ , edge_to, edge_from, a_to, a_from,probs,_, _ = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            probs = probs.to(device=self.DEVICE, dtype=torch.float32)

            # outputs = self.model(edge_to, edge_from)
            umap_l, new_l, recon_l, temporal_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, self.model, probs )
            all_loss.append(loss.mean().item())
            new_losses.append(new_l.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            temporal_losses.append(temporal_l.mean().item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.mean().backward()
            # loss_new.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('umap:{:.4f}\trecon_l:{:.4f}\tnew_loss:{:.4f}\tloss:{:.4f}'.format(sum(umap_losses) / len(umap_losses),
                                                                sum(recon_losses) / len(recon_losses),
                                                                sum(new_losses) / len(new_losses),
                                                                sum(all_loss) / len(all_loss)))
        return self.loss
    
    def train(self, PATIENT, MAX_EPOCH_NUMS, data_provider, iteration):
        patient = PATIENT
        time_start = time.time()
        for epoch in range(MAX_EPOCH_NUMS):
            print("====================\nepoch:{}\n===================".format(epoch+1))
            prev_loss = self.loss
            loss = self.train_step(data_provider, iteration,epoch)
            self.lr_scheduler.step()
            # early stop, check whether converge or not
            if prev_loss - loss < 5E-3:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

        time_end = time.time()
        time_spend = time_end - time_start
        print("Time spend: {:.2f} for training vis model...".format(time_spend))
    
    
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)

class BaseTextTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader,DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
    
    
    def train_step(self,data_provider,iteration,epoch):

        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        temporal_losses = []
        new_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))


        for data in t:
            _,_ , edge_to, edge_from, a_to, a_from,probs,_, _ = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            probs = probs.to(device=self.DEVICE, dtype=torch.float32)

            # outputs = self.model(edge_to, edge_from)
            umap_l, new_l, recon_l, temporal_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, self.model, probs )
            all_loss.append(loss.mean().item())
            new_losses.append(new_l.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            temporal_losses.append(temporal_l.mean().item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.mean().backward()
            # loss_new.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('umap:{:.4f}\trecon_l:{:.4f}\tnew_loss:{:.4f}\tloss:{:.4f}'.format(sum(umap_losses) / len(umap_losses),
                                                                sum(recon_losses) / len(recon_losses),
                                                                sum(new_losses) / len(new_losses),
                                                                sum(all_loss) / len(all_loss)))
        return self.loss
    
    def train(self, PATIENT, MAX_EPOCH_NUMS, data_provider, iteration):
        patient = PATIENT
        time_start = time.time()
        for epoch in range(MAX_EPOCH_NUMS):
            print("====================\nepoch:{}\n===================".format(epoch+1))
            prev_loss = self.loss
            loss = self.train_step(data_provider, iteration,epoch)
            self.lr_scheduler.step()
            # early stop, check whether converge or not
            if prev_loss - loss < 5E-3:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

        time_end = time.time()
        time_spend = time_end - time_start
        print("Time spend: {:.2f} for training vis model...".format(time_spend))
    
    
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)

class VISTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader,DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
    
    
    def train_step(self,data_provider,iteration,epoch,ifFreeze, interval):
        
        projector = PROCESSProjector(self.model, data_provider.content_path, '', self.DEVICE)
        evaluator = Evaluator(data_provider, projector)
        evaluator.eval_inv_train(iteration)
        evaluator.eval_inv_test(iteration)

        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        if ifFreeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            if interval:
                # Determine component to freeze based on whether the epoch is even or odd
                component_to_freeze = 'decoder' if epoch % 2 == 0 else 'encoder'
                print("interval freeze: {}".format(component_to_freeze))
                for name, param in self.model.named_parameters():
                    if component_to_freeze in name:
                        param.requires_grad = False
                    # print(f"Freezing {component_to_freeze}")
            else:
                print("freeze decoder only")
                # Default behavior (original freezing logic can be placed here)
                for name, param in self.model.named_parameters():
                    if 'decoder' in name:
                        # print("freezed")
                        param.requires_grad = False
        all_loss = []
        umap_losses = []
        recon_losses = []
        temporal_losses = []
        new_losses = []
        
        

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))

        train_data = data_provider.train_representation(iteration)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
  

        recon_train_data = self.model(torch.Tensor(train_data).to(self.DEVICE), torch.Tensor(train_data).to(self.DEVICE))['recon'][0]
        recon_pred = data_provider.get_pred(iteration, recon_train_data.detach().cpu().numpy())
        # if ifFreeze:
        #     vis = visualizer(data_provider, projector, 200, "tab10")
        #     conf_error,neg_grids,pos_grids = get_confidence_error_pairs(data_provider,iteration,projector,vis,0.2)

        for data in t:
            edge_to_idx, edge_from_idx, edge_to, edge_from, a_to, a_from,probs,pred_edge_to, pred_edge_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            probs = probs.to(device=self.DEVICE, dtype=torch.float32)

            pred_edge_to = pred_edge_to.to(device=self.DEVICE, dtype=torch.float32)
            pred_edge_from = pred_edge_from.to(device=self.DEVICE, dtype=torch.float32)

            recon_pred_edge_to = torch.Tensor(recon_pred[edge_to_idx]).to(device=self.DEVICE, dtype=torch.float32)
            recon_pred_edge_from = torch.Tensor(recon_pred[edge_from_idx]).to(device=self.DEVICE, dtype=torch.float32)

            # outputs = self.model(edge_to, edge_from)
            umap_l, new_l, recon_l, temporal_l, loss = self.criterion(edge_to_idx, edge_from_idx,edge_to, edge_from, a_to, a_from, self.model, probs,pred_edge_to, pred_edge_from,recon_pred_edge_to,recon_pred_edge_from,epoch )
            all_loss.append(loss.mean().item())
            new_losses.append(new_l.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            temporal_losses.append(temporal_l.mean().item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.mean().backward()
            # loss_new.backward()
            self.optimizer.step()
        self._loss = sum(all_loss) / len(all_loss)
        self.model.eval()
        print('umap:{:.4f}\trecon_l:{:.4f}\tnew_loss:{:.4f}\tloss:{:.4f}'.format(sum(umap_losses) / len(umap_losses),
                                                                sum(recon_losses) / len(recon_losses),
                                                                sum(new_losses) / len(new_losses),
                                                                sum(all_loss) / len(all_loss)))
        return self.loss
    
    def train(self, PATIENT, MAX_EPOCH_NUMS, data_provider, iteration, ifFreeze=False, interval=False):
        patient = PATIENT
        time_start = time.time()
        for epoch in range(MAX_EPOCH_NUMS):
            print("====================\nepoch:{}\n===================".format(epoch+1))
            prev_loss = self.loss
            loss = self.train_step(data_provider, iteration,epoch, ifFreeze, interval)
            self.lr_scheduler.step()
            # early stop, check whether converge or not
            if prev_loss - loss < 5E-3:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

        time_end = time.time()
        time_spend = time_end - time_start
        print("Time spend: {:.2f} for training vis model...".format(time_spend))
    
    
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)

    def calc_gradient(self,data_provider,iteration,core_epoch):
        self.model = self.model.to(device=self.DEVICE)
        self.model.eval()

        t = tqdm(self.edge_loader, leave=False, total=len(self.edge_loader))

        train_data = data_provider.train_representation(iteration)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])

        recon_train_data = self.model(torch.Tensor(train_data).to(self.DEVICE), torch.Tensor(train_data).to(self.DEVICE))['recon'][0]
        recon_pred = data_provider.get_pred(iteration, recon_train_data.detach().cpu().numpy())
        gradients_to = torch.zeros_like(torch.Tensor(train_data).to(self.DEVICE))
        gradients_from = torch.zeros_like(torch.Tensor(train_data).to(self.DEVICE))
        for data in t:
            self.optimizer.zero_grad()
            edge_to_idx, edge_from_idx, edge_to, edge_from, a_to, a_from,probs,pred_edge_to, pred_edge_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            probs = probs.to(device=self.DEVICE, dtype=torch.float32)

            pred_edge_to = pred_edge_to.to(device=self.DEVICE, dtype=torch.float32)
            pred_edge_from = pred_edge_from.to(device=self.DEVICE, dtype=torch.float32)

            recon_pred_edge_to = torch.Tensor(recon_pred[edge_to_idx]).to(device=self.DEVICE, dtype=torch.float32)
            recon_pred_edge_from = torch.Tensor(recon_pred[edge_from_idx]).to(device=self.DEVICE, dtype=torch.float32)

            umap_l, new_l, recon_l, temporal_l, loss = self.criterion(edge_to_idx, edge_from_idx,edge_to, edge_from, a_to, a_from, self.model, probs,pred_edge_to, pred_edge_from,recon_pred_edge_to,recon_pred_edge_from,core_epoch )
            with torch.no_grad():
                grads_to = torch.autograd.grad(loss, edge_to, retain_graph=True)[0]
                grads_from = torch.autograd.grad(loss, edge_from, retain_graph=True)[0]
                for i in range(len(edge_to_idx)):
                    gradients_to[edge_to_idx[i]] += grads_to[i]
                    gradients_from[edge_from_idx[i]] += grads_from[i]

        for idx in range(len(train_data)):
            count_to = torch.sum(edge_to_idx == idx).item()
            count_from = torch.sum(edge_from_idx == idx).item()
            if count_to > 0:
                gradients_to[idx] /= count_to
            if count_from > 0:
                gradients_from[idx] /= count_from
        gradients = (gradients_to + gradients_from) / 2
        self.model.train()
        return gradients
