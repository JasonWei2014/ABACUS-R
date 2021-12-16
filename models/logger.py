import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
# from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, loss_freq=100, zfill_num=8, log_file_name='log.txt', split="train"):

        self.loss_list = []
        self.split = split
        self.cpk_dir = log_dir
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        self.zfill_num = zfill_num
        self.checkpoint_freq = checkpoint_freq
        self.loss_freq = loss_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, model=None, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        if model is not None:
            model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.writer.close()

    def log_iter(self, losses, global_step):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))
        if global_step % self.loss_freq == 0:
            for name in self.names:
                self.writer.add_scalar('train/%s' % name, losses[name], global_step=global_step)

    def log_epoch(self, epoch, models):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
