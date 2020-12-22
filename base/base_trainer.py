# trainer class
import datetime
import json
import logging
import os
import time

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from base import DataPrefetcher
from utils import helpers
from utils import transforms as local_transforms
from utils.losses import reconstruction_loss
from utils.metrics import AverageMeter


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Trainer:
    def __init__(self, model, config, train_loader, train_logger=None, prefetch=True):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_decoder_params())},
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                     'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p: p.requires_grad, self.model.get_backbone_params()),
                                     'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

        # CHECKPOINTS
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.train_loader.batch_size) + 1
        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.device == torch.device('cpu'):
            prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
        torch.backends.cudnn.benchmark = True

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # RUN TRAIN
            results = self._train_epoch(epoch)

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (image, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            hidden, image_rec = self.model(image)
            # assert output.size()[2:] == target.size()[1:]
            # assert output.size()[1] == self.num_classes
            loss = reconstruction_loss(image, image_rec)

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # FOR EVAL
            # segmnetation = get_segmentation(output, target)
            # seg_metrics = eval_metrics(output, target, self.num_classes)
            # self._update_seg_metrics(*seg_metrics)
            # pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                epoch, self.total_loss.average,
                0, 0,
                self.batch_time.average, self.data_time.average))

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average}
        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
