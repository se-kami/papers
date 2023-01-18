#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from optimizer import Adam_impl


def make_fake_regr_data(h=None, d=4, size=10000, noise=0.1):
    """
    h: weights
    d: dimensions
    size: number of examples
    noise: noise
    """
    # hypothesis function
    if h is None:
        h_min, h_max = -5, 5
        h = torch.rand(d) * (h_max - h_min) + h_min
    else:
        h = torch.tensor(h)
    # random x
    x_min, x_max = -10, 10
    x = torch.rand((size, d)) * (x_max - x_min) + x_min
    # y with noise
    y = x @ h + torch.rand(size) * noise
    return x, y


class RegrDataset(Dataset):
    def __init__(self, data):
        """
        data: (x, y)
        x: N_samples x d
        y: N_samples x 1
        """
        super().__init__()
        x, y = data
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x)


def train(config,
          val_every=1,
          save_every=0,
          path='runs',
          to_cli=True,
          to_cli_every=1,
          to_tensorboard=True,
          to_tensorboard_every=1,
          tensorboard_tag=None,
          ):
    """
    config: dict containing:
                model, -> model
                train_loader, -> train data loader
                test_loader, -> test data loader
                epochs, -> number of epochs to train for
                optimizer, -> optimizer
                criterion, -> loss function
                device, -> device
    and optionally:
        scheduler -> None if not provided
    val_every: validation step frequency, increase for faster training, default 1
    name: model name, default model.pt
    path: directory for saving models and logs, default runs
    to_cli: whether to print training progress, defalt True
    to_cli_every: output every to_cli_epochs, default 1
    to_tensorboard: whether to save tensorboard logs, default False
    to_tensorboard_every: save frequency, default 1
    logger: if not None, use that logger to track progress, default None
    logger_every: save frequency, default 1
    """
    # load
    model = config['model']
    train_loader = config['train_loader']
    test_loader = config['test_loader']
    epochs = config['epochs']
    optimizer = config['optimizer']
    criterion = config['criterion']
    device = config['device']
    scheduler = config['scheduler'] if 'scheduler' in config else None

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    epoch_len = len(str(epochs))

    # make save dir
    if path is not None:
        os.makedirs(path, exist_ok=True)

    # init variables
    train_losses, test_losses = [], []
    test_loss_min = np.Inf
    start_time = time.time()

    # tensorboard init
    if to_tensorboard:
        if path is not None:
            tag = path.split('/')[-1]
        else:
            tag = str(time.time()).split('.')[0]
        log_dir = f"runs/tensorboard/{tag}"
        if tensorboard_tag:
            log_dir += f'/{tensorboard_tag}'
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('Learning rate', config['lr'], 0)
        writer.add_text('summary', config.get('summary', ''), 0)

    # train loop
    for epoch in range(epochs):
        # init variables
        train_loss, test_loss = 0.0, 0.0
        epoch_start_time = time.time()
        epoch_str = f"{epoch+1:{epoch_len}}/{epochs:{epoch_len}}"

        model.train()

        # train
        if to_cli and epoch % to_cli_every == 0:
            print(f"epoch: {epoch_str} starting training")
        train_start_time = time.time()
        for xb, yb in tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)  # move to device
            optimizer.zero_grad()  # zero gradient
            yp = model(xb)  # forward pass
            loss = criterion(yp, yb)  # loss
            loss.backward()  # backward pass
            optimizer.step()  # optimizer step


            # log losses
            train_loss += loss.item() * xb.size(0)

        train_total_time = time.time() - train_start_time
        train_loss = train_loss/train_size  # average loss over epoch
        train_losses.append(train_loss * 100)  # log


        if to_cli and epoch % to_cli_every == 0:
            print(f"epoch: {epoch_str} training done in {train_total_time:.1f} s")
            print(f"epoch: {epoch_str} training loss: {train_loss}")

        if to_tensorboard and epoch % to_tensorboard_every == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)

        # test
        if epoch % val_every == 0:
            if to_cli and epoch % to_cli_every == 0:
                print(f"epoch: {epoch_str} starting test")

            test_start_time = time.time()
            model.eval()
            with torch.no_grad():
                for xb, yb in tqdm(test_loader):
                    xb, yb = xb.to(device), yb.to(device)
                    yp = model(xb)
                    loss = criterion(yp, yb)
                    test_loss += loss.item() * xb.size(0)

            test_total_time = time.time() - test_start_time
            test_loss = test_loss / test_size  # average over epoch
            test_losses.append(test_loss * 100)  # log

            if to_cli and epoch % to_cli_every == 0:
                print(f"epoch: {epoch_str} test done in {test_total_time:.1f} s")
                print(f"epoch: {epoch_str} test loss: {test_loss}")

            if to_tensorboard and epoch % to_tensorboard_every == 0:
                writer.add_scalar('Loss/test', test_loss, epoch)

            # save model if the performance improved
            if test_loss <= test_loss_min and path is not None:
                if to_cli:
                    print(f"Loss decreased from {test_loss_min} to {test_loss}")
                    print(f"Saving model")
                name_best = f"{path}/model-best-{epoch+1:0{epoch_len}}.pth"
                torch.save(model.state_dict(), name_best)
                test_loss_min = test_loss

        if scheduler is not None:
            scheduler.step(test_loss)  # scheduler step
        # time taken for one epoch
        time_total = time.time() - epoch_start_time
        if to_cli and epoch % to_cli_every == 0:
            h = int(time_total // 3600)
            m = int(time_total // 60 - h*60)
            s = time_total % 60
            print(f"epoch: {epoch_str} completed in {h:02}:{m:02}:{s:04.1f}")
        print('\n\n')
        if save_every > 0 and epoch % save_every == 0:
                name = f"{path}/model-{epoch+1:0{epoch_len}}.pth"
                torch.save(model.state_dict(), name)

    # total time
    time_total = time.time() - start_time
    if to_cli:
        h = int(time_total // 3600)
        m = int(time_total // 60 - h*60)
        s = time_total % 60
        print(f"Training completed in {h:02}:{m:02}:{s:04.1f}")

    # save metrics
    metric_dict = dict()
    metric_dict['train_loss'] = train_losses
    metric_dict['test_loss'] = test_losses
    df = pd.DataFrame(metric_dict)
    df.to_csv(f'{path}/results.csv')


if __name__ == '__main__':
    d = 10
    data = make_fake_regr_data(d=d, size=1000*10)
    ds = RegrDataset(data)
    train_size = int(0.8*len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)
    criterion = torch.nn.MSELoss()

    config = dict()
    config['train_loader'] = train_loader
    config['test_loader'] = test_loader
    config['epochs'] = 50
    config['device'] = torch.device('cpu')
    config['lr'] = 0.01
    config['criterion'] = nn.MSELoss()

    optimizers = []
    optimizers.append([Adam_impl, 'Adam_impl'])  # implemented Adam
    optimizers.append([torch.optim.Adam, 'Adam'])  # torch Adam
    optimizers.append([torch.optim.SGD, 'SGD'])  # SGD

    for optimizer, summary in optimizers:
        model = Net(d)
        config['model'] = model
        config['optimizer'] = optimizer(config['model'].parameters(), lr=config['lr'])
        config['summary'] = summary
        train(config, tensorboard_tag=config['summary'])
