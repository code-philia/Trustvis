

import torch
import numpy as np



class SkeletonGenerator:
    """SkeletonGenerator except allows for generate skeleton"""
    def __init__(self, data_provider, epoch, interval=25,base_num_samples=100):
        """
        interval: int : layer number of the radius
        """
        self.data_provider = data_provider
        self.epoch = epoch
        self.interval = interval
        self.base_num_samples= base_num_samples
       
    def skeleton_gen(self):
        torch.manual_seed(0)  # freeze the radom seed
        torch.cuda.manual_seed_all(0)

        # Set the random seed for numpy
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train_data=self.data_provider.train_representation(epoch=self.epoch)
        train_data = torch.Tensor(train_data)
        center = train_data.mean(dim=0)
        # calculate the farest distance
        radius = ((train_data - center)**2).sum(dim=1).max().sqrt()
        print("radius,radius",radius)

        min_radius_log = np.log10(1e-3)
        max_radius_log = np.log10(radius.item() * 1.1)
        # *****************************************************************************************
        # generate 100 points in log space 
        radii_log = np.linspace(max_radius_log, min_radius_log, self.interval)
        # convert back to linear space
        radii = 10 ** radii_log

    
        # calculate the number of samples for each radius
        num_samples_per_radius_l = []
        for r in radii:
            # calculate the log surface area for the current radius
            # convert it back to the original scale
            # calculate the number of samples
            num_samples = int(self.base_num_samples * r // 4)
            num_samples_per_radius_l.append(num_samples)
        

       

        # *****************************************************************************************

        radii = [radius*1.1, radius, radius / 2, radius / 4, radius / 10, 1e-3]  # radii at which to sample points
        # num_samples_per_radius_l = [500, 500, 500, 500, 500, 500]  # number of samples per radius
        aaa = 2000
        num_samples_per_radius_l = [aaa, aaa, aaa, aaa, aaa, aaa]  # number of samples per radius
        print("num_samples_per_radius_l",radii)
        print("num_samples_per_radius_l",num_samples_per_radius_l)
        # list to store samples at all radii
        high_bom_samples = []

        for i in range(len(radii)):
            r = radii[i]

            num_samples_per_radius = num_samples_per_radius_l[i]
            # sample points on the sphere with radius r
            samples = torch.randn(num_samples_per_radius, 512)
            samples = samples / samples.norm(dim=1, keepdim=True) * r

            high_bom_samples.append(samples)

            # concatenate samples from all radii
            high_bom = torch.cat(high_bom_samples, dim=0)

            high_bom = high_bom.cpu().detach().numpy()
        
        return high_bom

    