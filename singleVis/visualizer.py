from abc import ABC, abstractmethod

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import base64
import json

class VisualizerAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider, projector, * args, **kawargs):
        pass

    @abstractmethod
    def _init_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_epoch_plot_measures(self, *args, **kwargs):
        # return x_min, y_min, x_max, y_max
        pass

    @abstractmethod
    def get_epoch_decision_view(self, *args, **kwargs):
        pass

    @abstractmethod
    def savefig(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_background(self, *args, **kwargs):
        pass

    @abstractmethod
    def show_grid_embedding(self, *args, **kwargs):
        pass

class visualizer(VisualizerAbstractClass):
    def __init__(self, data_provider, projector, resolution, cmap='tab10'):
        self.data_provider = data_provider
        self.projector = projector
        self.cmap = plt.get_cmap(cmap)
        self.classes = data_provider.classes
        self.class_num = len(self.classes)
        self.resolution= resolution

    def _init_plot(self, only_img=False):
        '''
        Initialises matplotlib artists and plots. from DeepView and DVI
        '''
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        if not only_img:
            self.ax.set_title("TimeVis visualization")
            self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
            self.ax.legend()
        else:
            self.ax.set_axis_off()
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []
        # labels = prediction
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=10,
                mec='black', mew=0.5,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])
            
        # labels != prediction, labels be a large circle
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color, 
                fillstyle='full', ms=7, mew=2.5, zorder=3)
            self.sample_plots.append(plot[0])

        # labels != prediction, prediction stays inside of circle
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], '.', markeredgecolor=color,
                                fillstyle='full', ms=6, zorder=4)
            self.sample_plots.append(plot[0])
        
        # Initialize white border points
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], '.', label="border", ms=5,
                    color="yellow", markeredgecolor=color, zorder=6, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        color = (0.0, 0.0, 0.0, 1.0)
        plot = self.ax.plot([], [], '.', markeredgecolor=color,
                            fillstyle='full', ms=20, zorder=1)
        self.sample_plots.append(plot[0])
        self.disable_synth = False
    
    def _init_default_plot(self, only_img=True):
        '''
        Initialises matplotlib artists and plots. from DeepView and DVI
        '''
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        if not only_img:
            self.ax.set_title("TimeVis visualization")
            self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
            self.ax.legend()
        else:
            self.ax.set_axis_off()
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=5,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])
        self.disable_synth = False
        
    
    def get_epoch_plot_measures(self, epoch):
        """get plot measure for visualization"""
        data = self.data_provider.train_representation(epoch)
        data = data.reshape(data.shape[0], data.shape[1])
        
        embedded = self.projector.batch_project(epoch, data)

        ebd_min = np.min(embedded, axis=0)
        ebd_max = np.max(embedded, axis=0)
        ebd_extent = ebd_max - ebd_min

        x_min, y_min = ebd_min - 0.1 * ebd_extent
        x_max, y_max = ebd_max + 0.1 * ebd_extent

        x_min = min(x_min, y_min)
        y_min = min(x_min, y_min)
        x_max = max(x_max, y_max)
        y_max = max(x_max, y_max)

        

        return x_min, y_min, x_max, y_max
    
    def get_grid(self, epoch, resolution, xy_limit=None):
        if xy_limit is None:
            x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)
        else:
            x_min, y_min, x_max, y_max = xy_limit
        # create grid
        
        print([x_min, y_min, x_max, y_max])
        save_dir = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch))
        scale_path = os.path.join(save_dir, "scale.npy")
        np.save(scale_path, [x_min, y_min, x_max, y_max])
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(grid.reshape(grid.shape[0], -1), 0, 1)
        return grid

    
    def get_epoch_decision_view(self, epoch, resolution, xy_limit=None, forDetail=False):
        '''
        get background classifier view
        :param epoch_id: epoch that need to be visualized
        :param resolution: background resolution
        :return:
            grid_view : numpy.ndarray, self.resolution,self.resolution, 2
            decision_view : numpy.ndarray, self.resolution,self.resolution, 3
        '''
        print('Computing decision regions ...')
        grid = self.get_grid(epoch, resolution, xy_limit)

        # map gridmpoint to images
        grid_samples = self.projector.batch_inverse(epoch, grid)
        print("grid_samples",grid_samples.shape)

        mesh_preds = self.data_provider.get_pred(epoch, grid_samples)
        mesh_preds = mesh_preds + 1e-8

        sort_preds = np.sort(mesh_preds, axis=1)
        diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        
        border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        border[diff < 0] = 1
        diff[border == 1] = 0.

        diff = diff/(diff.max()+1e-8)
        diff = diff*0.6

        mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(mesh_classes)
        color = self.cmap(mesh_classes / mesh_max_class)

        diff = diff.reshape(-1, 1)

        color = color[:, 0:3]
        inint_color = color

        color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        for i, c in enumerate(diff):
            # if c< 0.2 and c > 0.19:  # 当置信度 >= 0.8时，
            # if c < 0.005:
            #     print("init_color")
            #     color[i] = np.ones(inint_color[i], dtype=np.uint8)
            if c< 0.15 and c > 0.01:
                d = 0.2
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            if c< 0.2 and c > 0.15:
                d = 0.4
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.3 and c >= 0.2:  # 当置信度 >= 0.8时，
                d = 0.5
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.4 and c >= 0.3:  # 当置信度 >= 0.8时，颜色趋向黑色
                d = 0.6
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.5 and c >= 0.4:  # 当置信度 >= 0.8时，颜色趋向黑色
                d = 0.7
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.6 and c >= 0.5:  # 当置信度 >= 0.8时，颜色趋向黑色
                d = 0.8
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.7 and c >= 0.6:  # 当置信度 >= 0.8时，颜色趋向黑色
                d = 0.9
                color[i] = d * 0.5 * inint_color[i] + (1 - d) * np.ones(inint_color[i].shape, dtype=np.uint8)  # 黑色
            elif c< 0.8 and c >= 0.7:  # 当置信度 >= 0.8时，颜色趋向黑色
                color[i] = inint_color[i]  # 黑色
            # elif c< 1 and c > 0.99:  # 当置信度 >= 0.8时，颜色趋向黑色
            #     color[i] = inint_color[i]  # 黑色
        decision_view = color.reshape(resolution, resolution, 3)
        grid_view = grid.reshape(resolution, resolution, 2)
        if forDetail == True:
            return grid_samples, grid, border
        
        return grid_view, decision_view
    
    def get_epoch_decision_view_text(self, epoch, resolution, xy_limit=None, forDetail=False):
        '''
        get background classifier view
        :param epoch_id: epoch that need to be visualized
        :param resolution: background resolution
        :return:
            grid_view : numpy.ndarray, self.resolution,self.resolution, 2
            decision_view : numpy.ndarray, self.resolution,self.resolution, 3
        '''
        print('Computing decision regions ...')

        if xy_limit is None:
            x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)
        else:
            x_min, y_min, x_max, y_max = xy_limit


        print([x_min, y_min, x_max, y_max])
        save_dir = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch))
        scale_path = os.path.join(save_dir, "scale.npy")
        np.save(scale_path, [x_min, y_min, x_max, y_max])

    def save_scale_bgimg(self, epoch):
        
        '''
        Shows the current plot.
        '''
        self._init_plot(only_img=True)
        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)
        print([x_min, y_min, x_max, y_max])
        save_dir = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch))
        scale_path = os.path.join(save_dir, "scale.npy")
        np.save(scale_path, [x_min, y_min, x_max, y_max])
        from PIL import Image
        img = Image.new("RGB",(200,200),(255,255,255))
        bgimg_path = os.path.join(self.data_provider.model_path, "Epoch_{}".format(epoch), "bgimg.png")

        img.save(bgimg_path)
        
    
    def savefig(self, epoch, path="vis", indicates=[]):
        
        '''
        Shows the current plot.
        '''
        self._init_plot(only_img=True)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        # with torch.no_grad():
            # mu, _ = self.projector.
            # mu = mu.cpu().numpy()  # Convert to numpy array for easier manipulation

            # Xmin, Ymin = mu.min(axis=0)  # Minimum values for each dimension
            # Xmax, Ymax = mu.max(axis=0)  # Maximum values for each dimension



        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        # params_str = 'res: %d'
        # desc = params_str % (self.resolution)
        # self.desc.set_text(desc)

        # train_labels = self.data_provider.train_labels(epoch)
        
        # indices = np.where(train_labels == 2)

        train_data = self.data_provider.train_representation(epoch)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
        train_labels = self.data_provider.train_labels(epoch)
        pred = self.data_provider.get_pred(epoch, train_data)
        pred = pred.argmax(axis=1)
        
        

        embedding = self.projector.batch_project(epoch, train_data)
        
        if len(indicates):
            embedding = embedding[indicates]
            pred = pred[indicates]
            train_data = train_data[indicates]
            train_labels = train_labels[indicates]
             
            

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, train_labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)
        
    def savefig_custom(self, epoch, path="vis", embedding=[],pred=[],train_labels=[]):
        
        '''
        Shows the current plot.
        '''
        self._init_plot(only_img=True)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        # with torch.no_grad():
            # mu, _ = self.projector.
            # mu = mu.cpu().numpy()  # Convert to numpy array for easier manipulation

            # Xmin, Ymin = mu.min(axis=0)  # Minimum values for each dimension
            # Xmax, Ymax = mu.max(axis=0)  # Maximum values for each dimension

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        # params_str = 'res: %d'
        # desc = params_str % (self.resolution)
        # self.desc.set_text(desc)

        # train_labels = self.data_provider.train_labels(epoch)
        
        # indices = np.where(train_labels == 2)
            

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, train_labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)
    

    def show_grid_embedding(self, epoch, data, embedding, border, noOutline=False, path="vis"):
        '''
        Shows the current plot.
        '''
        self._init_plot(only_img=True)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        # params_str = 'res: %d'
        # desc = params_str % (self.resolution)
        # self.desc.set_text(desc)
        train_labels = self.data_provider.get_pred(epoch, data)
        train_labels = train_labels.argmax(axis=1)
        
        inv = self.projector.batch_inverse(epoch, embedding)
        pred = self.data_provider.get_pred(epoch, inv)
        pred = pred.argmax(axis=1)
        

        # mesh_preds = self.data_provider.get_pred(epoch, inv)
        # mesh_preds = mesh_preds + 1e-8

        # sort_preds = np.sort(mesh_preds, axis=1)
        # diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        # border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        # border[diff < 0.15] = 1

        # mesh_preds = self.data_provider.get_pred(epoch, data) + 1e-8

        # sort_preds = np.sort(mesh_preds, axis=1)
        # diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        # border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        # border[diff < 0.15] = 1




        if noOutline == True:
            for c in range(self.class_num):
                data = embedding[np.logical_and(train_labels == c, border!=1)]
                self.sample_plots[c].set_data(data.transpose())
            for c in range(self.class_num):
                data = embedding[np.logical_and(train_labels == c, border==1)]
                self.sample_plots[3*self.class_num + c].set_data(data.transpose())
        else: 
            for c in range(self.class_num):
                data = embedding[np.logical_and(train_labels == c, train_labels == pred, border!=1)]
                self.sample_plots[c].set_data(data.transpose())
            for c in range(self.class_num):
                data = embedding[np.logical_and(train_labels == c, train_labels != pred, border!=1)]
                self.sample_plots[self.class_num+c].set_data(data.transpose())
            for c in range(self.class_num):
                data = embedding[np.logical_and(pred == c, train_labels != pred, border!=1)]
                self.sample_plots[2*self.class_num + c].set_data(data.transpose())
            for c in range(self.class_num):
                data = embedding[np.logical_and(train_labels == c, border==1)]
                self.sample_plots[3*self.class_num+ c].set_data(data.transpose())



        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)
    
    def save_default_fig(self, epoch, path="vis"):
        '''
        Shows the current plot.
        '''
        self._init_default_plot(only_img=True)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        train_data = self.data_provider.train_representation(epoch)
        train_labels = self.data_provider.train_labels(epoch)
        pred = self.data_provider.get_pred(epoch, train_data)
        pred = pred.argmax(axis=1)

        embedding = self.projector.batch_project(epoch, train_data)

        for c in range(self.class_num):
            data = embedding[train_labels == c]
            self.sample_plots[c].set_data(data.transpose())
        plt.savefig(path)
    
    def savefig_cus(self, epoch, data, pred, labels, path="vis"):
        '''
        Shows the current plot with given data
        '''
        self._init_plot(only_img=True)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        # params_str = 'res: %d'
        # desc = params_str % (self.resolution)
        # self.desc.set_text(desc)
        embedding = self.projector.batch_project(epoch, data)

        for c in range(self.class_num):
            data = embedding[np.logical_and(labels == c, labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(labels == c, labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.savefig(path)

    
    def savefig_trajectory(self, epoch, xs, ys, xy_limit=None, path="vis"):
        '''
        Shows the current plot with given data
        '''
        self._init_plot(only_img=True)

        if xy_limit is None:
            x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)
        else:
            x_min, y_min, x_max, y_max = xy_limit

        _, decision_view = self.get_epoch_decision_view(epoch, self.resolution, xy_limit)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        self.sample_plots[-1].set_data(np.vstack((xs,ys)))

        # set data point
        u = xs[1:] - xs[:-1]
        v = ys[1:] - ys[:-1]

        x = xs[:len(u)] # 使得维数和u,v一致
        y = ys[:len(v)]

        # plt.quiver(prev_embedding[:, 0], prev_embedding[:, 1], embedding[:, 0]-prev_embedding[:, 0],embedding[:, 1]-prev_embedding[:, 1], scale_units='xy', angles='xy', scale=1, color='black')  
        plt.quiver(x,y,u,v, angles='xy', scale_units='xy', scale=1, color="black")
        plt.savefig(path)
    
    def get_background(self, epoch, resolution):
        '''
        Initialises matplotlib artists and plots. from DeepView and DVI
        '''
        plt.ion()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(1, 1, figsize=(200*px, 200*px))
        ax.set_axis_off()
        cls_plot = ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)
        # self.disable_synth = False
        fname = "Epoch" if self.data_provider.mode == "normal" else "Iteration"

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch)
        scale_path =  os.path.join(self.data_provider.model_path, "{}_{}".format(fname, epoch), "scale.npy")
        np.save(scale_path,[x_min, y_min, x_max, y_max])
        _, decision_view = self.get_epoch_decision_view(epoch, resolution)

        cls_plot.set_data(decision_view)
        cls_plot.set_extent((x_min, x_max, y_max, y_min))
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

        # save first and them load
        
        save_path = os.path.join(self.data_provider.model_path, "{}_{}".format(fname, epoch), "bgimg.png")
        plt.savefig(save_path, format='png',bbox_inches='tight',pad_inches=0.0)
        with open(save_path, 'rb') as img_f:
            img_stream = img_f.read()
            save_file_base64 = base64.b64encode(img_stream)
    
        return x_min, y_min, x_max, y_max, save_file_base64
    
    def get_standard_classes_color(self):
        '''
        get the RGB value for 10 classes
        :return:
            color : numpy.ndarray, shape (10, 3)
        '''
        # TODO 10 classes?
        mesh_max_class = self.class_num - 1
        mesh_classes = np.arange(len(self.classes))
        color = self.cmap(mesh_classes / mesh_max_class)
        color = color[:, 0:3]
        # color = np.concatenate((color, np.zeros((1,3))), axis=0)
        return color
