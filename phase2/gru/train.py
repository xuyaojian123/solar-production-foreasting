#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : train1.py
@Author: XuYaoJian
@Date  : 2022/11/1 16:23
@Desc  : 
"""
from data_processed import *
from prepare import prep_env
from models import GruModel
from torch.utils.data import DataLoader
from dataset import TrainDataset
import torch
from torch import nn
import time
import os
import random

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.cuda.set_device(device)


def seed_everything(seed=2222):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(2222)
    settings = prep_env()

    for horizon in settings['horizons']:
        settings["output_len"] = horizon

        train_features, train_targets = get_data(settings, "../data/origin.csv")
        train_dataset = TrainDataset(train_features, train_targets, settings)
        train_dataloader = DataLoader(train_dataset, batch_size=settings['batch_size'], shuffle=True)

        model = GruModel(settings).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        criterion = nn.MSELoss(reduction='mean')
        steps_per_epoch = len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6, verbose=True)

        model.train()
        for epoch in range(settings['epoch_num']):
            scheduler.step()
            train_loss = 0
            t = time.time()
            for step, batch in enumerate(train_dataloader):
                features, targets = batch
                features = features.to(device)

                targets = targets[:, -settings['output_len']:].to(device)

                optimizer.zero_grad()
                output = model(features)

                loss = criterion(output, targets)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()

            print("horizons:{} epoch {}, Loss: {:.3f} Time: {:.1f}s"
                  .format(horizon, epoch + 1, train_loss / steps_per_epoch, time.time() - t))
        torch.save(model, settings['checkpoints'] + f"gru_o_60.pt")
        torch.cuda.empty_cache()
