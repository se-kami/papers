import torch
import numpy as np
import pandas as pd
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix, Accuracy
from plotutils import plot_confusion_matrix
from tqdm import tqdm


def train(config,
          val_every=1,
          save_every=0,
          path='runs',
          to_cli=True,
          to_cli_every=1,
          to_tensorboard=False,
          to_tensorboard_every=1,
          logger=None,
          logger_every=1,
          confusion_matrix_plot=False,
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
    confusion_matrix_plot: whether to make confusion matrix plot, default False
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
    classes = config['classes']
    num_classes = len(classes)
    if confusion_matrix_plot:
        confmat_fn = ConfusionMatrix(num_classes=num_classes)
    accuracy_fn = Accuracy()
    accuracy_w_fn = Accuracy(num_classes=num_classes, average='weighted')

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    epoch_len = len(str(epochs))

    # make save dir
    if path is not None:
        os.makedirs(path, exist_ok=True)

    # init variables
    train_losses, test_losses, train_accs, test_accs, train_accs_w, test_accs_w = [], [], [], [], [], []
    test_loss_min = np.Inf
    start_time = time.time()
    # save true labels once, used for accuracy
    labels_true_test = []
    for xb, yb in test_loader:
        labels_true_test.append(yb)
    labels_true_test = torch.cat(labels_true_test)

    # tensorboard init
    if to_tensorboard:
        if path is not None:
            tag = path.split('/')[-1]
        else:
            tag = str(time.time()).split('.')[0]
        log_dir = f"runs/tensorboard/{tag}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('Learning rate', config['lr'], 0)
        writer.add_text('summary', config.get('summary', ''), 0)

    # train loop
    for epoch in range(epochs):
        # init variables
        train_loss, test_loss = 0.0, 0.0
        epoch_start_time = time.time()
        predicted, labels_true_train = [], []
        if confusion_matrix_plot:
            confmat = torch.zeros((num_classes, num_classes))
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

            # accuracy
            _, predicted_y = torch.max(yp.data, 1)
            predicted.append(predicted_y.cpu())
            labels_true_train.append(yb.cpu())

        predicted = torch.cat(predicted)
        labels_true_train = torch.cat(labels_true_train)
        if confusion_matrix_plot:
            confmat = confmat_fn(predicted, labels_true_train).numpy()
            confmat_fig = plot_confusion_matrix(confmat, classes)
            confmat_fig.savefig(f'{path}/confmat-test-{epoch+1:0{epoch_len}}.png')

        train_total_time = time.time() - train_start_time
        train_loss = train_loss/train_size  # average loss over epoch
        train_losses.append(train_loss * 100)  # log

        train_acc = accuracy_fn(predicted, labels_true_train) * 100
        train_acc_w = accuracy_w_fn(predicted, labels_true_train) * 100
        train_accs.append(train_acc)
        train_accs_w.append(train_acc_w)

        if to_cli and epoch % to_cli_every == 0:
            print(f"epoch: {epoch_str} training done in {train_total_time:.1f} s")
            print(f"epoch: {epoch_str} training loss: {train_loss}")
            print(f"epoch: {epoch_str} training accuracy: {train_acc:.2f} %")
            print(f"epoch: {epoch_str} training weighted accuracy: {train_acc_w:.2f} %")

        if to_tensorboard and epoch % to_tensorboard_every == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            if confusion_matrix_plot:
                writer.add_figure('Confusion matrix/train', confmat_fig, epoch)

        # test
        if epoch % val_every == 0:
            if to_cli and epoch % to_cli_every == 0:
                print(f"epoch: {epoch_str} starting test")

            test_start_time = time.time()
            model.eval()
            predicted = []
            if confusion_matrix_plot:
                confmat = torch.zeros((num_classes, num_classes))
            with torch.no_grad():
                for xb, yb in tqdm(test_loader):
                    xb, yb = xb.to(device), yb.to(device)
                    yp = model(xb)
                    loss = criterion(yp, yb)
                    test_loss += loss.item() * xb.size(0)

                    # accuracy
                    _, predicted_y = torch.max(yp.data, 1)
                    predicted.append(predicted_y.cpu())

            predicted = torch.cat(predicted)
            if confusion_matrix_plot:
                confmat = confmat_fn(predicted, labels_true_test).numpy()
                confmat_fig = plot_confusion_matrix(confmat, classes)
                confmat_fig.savefig(f'{path}/confmat-test-{epoch+1:0{epoch_len}}.png')

            test_total_time = time.time() - test_start_time
            test_loss = test_loss / test_size  # average over epoch
            test_losses.append(test_loss * 100)  # log
            test_acc = accuracy_fn(predicted, labels_true_test) * 100
            test_accs.append(test_acc)
            test_acc_w = accuracy_w_fn(predicted, labels_true_test) * 100
            test_accs_w.append(test_acc_w)

            if to_cli and epoch % to_cli_every == 0:
                print(f"epoch: {epoch_str} test done in {test_total_time:.1f} s")
                print(f"epoch: {epoch_str} test loss: {test_loss}")
                print(f"epoch: {epoch_str} test accuracy: {test_acc:.2f} %")
                print(f"epoch: {epoch_str} test weighted accuracy: {test_acc_w:.2f} %")

            if to_tensorboard and epoch % to_tensorboard_every == 0:
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/test', test_acc, epoch)
                if confusion_matrix_plot:
                    writer.add_figure('Confusion matrix/test', confmat_fig, epoch)

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
    metric_dict['train_acc'] = train_accs
    metric_dict['test_acc'] = test_accs
    metric_dict['train_acc_w'] = train_accs_w
    metric_dict['test_acc_w'] = test_accs_w
    df = pd.DataFrame(metric_dict)
    df.to_csv(f'{path}/results.csv')


if True:
    from data import get_transform, get_data, get_img_to_tensor
    from device import get_device
    import pickle
    from config import config
    from model import EfficientNet

    train_transform = get_transform()
    test_transform = get_img_to_tensor()
    train_loader, test_loader, le, train_weights = get_data(
            config['data'],
            train_batch_size=config['train_batch_size'],
            test_batch_size=config['test_batch_size'],
            train_transform=train_transform,
            test_transform=test_transform,
            )
    with open('le.pkl', 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path = 'runs' + '/' + str(time.time()).split('.')[0]

    model = EfficientNet('B0', n_out=len(le))
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor(train_weights).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    summary = "EfficientNet"

    config['model'] = model
    config['device'] = device
    config['train_loader'] = train_loader
    config['test_loader'] = test_loader
    config['optimizer'] = optimizer
    config['criterion'] = criterion
    config['scheduler'] = scheduler
    config['classes'] = le
    config['summary'] = summary
    train(config, path=path, to_tensorboard=True)
