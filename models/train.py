from time import time
t000 = time()

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import trange

from logger import Logger

from torch.optim.lr_scheduler import MultiStepLR


def train(config, model, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    loss_weights = train_params["loss_weights"]

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, model, optimizer)
    else:
        start_epoch = 0

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=10, drop_last=True)
    print("dataloader")
    global_step = 0

    with Logger(log_dir=log_dir, checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for batch_idx, (label, cent_inf, knn_inf) in enumerate(dataloader):
                # t000 = time()
                # print(time() - t000)
                if torch.cuda.is_available():
                    label = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in label.items()}
                    cent_inf = {
                        'pdbname': cent_inf['pdbname'],
                        'node_dihedral': {key: value.cuda() for key, value in cent_inf['node_dihedral'].items()},
                        'dist': cent_inf['dist'].cuda()
                    }
                    knn_inf = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in knn_inf.items()}

                output = model(cent_inf, knn_inf)

                output = {
                    'logits': output[:, :20],
                    'bfactor': output[:, 20:21],
                    'ss3': output[:, 21:24],
                    'ss8': output[:, 24:32],
                    'rsa': output[:, 32:33],
                    'k1k2': output[:, 33:]
                }

                loss_values = {}

                if loss_weights['loss_AA'] != 0:
                    loss = F.cross_entropy(output['logits'], label['centralAA'])
                    loss_values['loss_AA'] = loss * loss_weights['loss_AA']

                if loss_weights['loss_bfactor'] != 0:
                    loss = (output['bfactor'].squeeze() - label['nodebfactor']).abs().mean()
                    loss_values['loss_bfactor'] = loss * loss_weights['loss_bfactor']

                if loss_weights['loss_ss3'] != 0:
                    loss = F.cross_entropy(output['ss3'], label['ss3'])
                    loss_values['loss_ss3'] = loss * loss_weights['loss_ss3']

                if loss_weights['loss_ss8'] != 0:
                    loss = F.cross_entropy(output['ss8'], label['ss8'])
                    loss_values['loss_ss8'] = loss * loss_weights['loss_ss8']

                if loss_weights['loss_rsa'] != 0:
                    loss = (output['rsa'].squeeze() - label['rsa']).abs().mean()
                    loss_values['loss_rsa'] = loss * loss_weights['loss_rsa']

                if loss_weights['loss_k1k2'] != 0:
                    loss = ((F.tanh(output['k1k2']) - label['k1k2']) * label['k1k2_mask']).abs().sum() / label['k1k2_mask'].sum()
                    loss_values['loss_k1k2'] = loss * loss_weights['loss_k1k2']

                loss = sum([val.mean() for val in loss_values.values()])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1

                loss_values = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_values.items()}
                logger.log_iter(losses=loss_values, global_step=global_step)

            scheduler.step()

            logger.log_epoch(epoch, {'model': model, 'optimizer': optimizer})
