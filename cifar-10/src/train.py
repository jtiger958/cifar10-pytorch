from model.VGG import vgg16
import torch
import os
from glob import glob
import torch.nn as nn
from visdom import Visdom

from utils.utils import AverageMeter, LambdaLR


class Trainer:
    def __init__(self, config, train_loader, num_class=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.num_class = num_class
        self.learning_rate = config.lr
        self.train_loader = train_loader
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.decay_epoch = config.decay_epoch
        self.loss_dir = config.loss_dir
        self.visdom = Visdom()

        self.build_model()

    def build_model(self):
        self.net = vgg16(self.num_class)
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        model = glob(os.path.join(self.checkpoint_dir, f"vgg16-{self.epoch-1}.pth"))

        if not model:
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            return

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print("[*] Load Model from %s: " % str(self.checkpoint_dir), str(model[-1]))

    def read_loss_info(self):
        train_loss_path = glob(os.path.join(self.loss_dir, "train_loss.txt"))
        learning_rate_path = glob(os.path.join(self.loss_dir, "learning_rate_info.txt"))
        epoch_info_path = glob(os.path.join(self.loss_dir, "epoch_info.txt"))

        if not train_loss_path:
            return [], [], []

        loss_file = open(train_loss_path[0], 'r')
        learning_rate_file = open(learning_rate_path[0], 'r')
        epoch_file = open(epoch_info_path[0], 'r')

        loss = loss_file.readline().split(' ')
        learning_rate = learning_rate_file.readline().split(' ')
        epoch = epoch_file.readline().split(' ')

        loss = [float(loss_item) for loss_item in loss[:-1]]
        learning_rate = [float(learning_rate_item) for learning_rate_item in learning_rate[:-1]]
        epoch = [int(epoch_item) for epoch_item in epoch[:-1]]

        loss_file.close()
        learning_rate_file.close()

        return loss, learning_rate, epoch

    def save_loss_info(self, loss, lr, epoch):
        train_loss_path = os.path.join(self.loss_dir, "train_loss.txt")
        learning_rate_path = os.path.join(self.loss_dir, "learning_rate_info.txt")
        epoch_info_path = os.path.join(self.loss_dir, "epoch_info.txt")

        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)

        loss_file = open(train_loss_path, 'w')
        learning_rate_file = open(learning_rate_path, 'w')
        epoch_info_file = open(epoch_info_path, 'w')

        for loss_item in loss:
            loss_file.write(f"{loss_item} ")

        for lr_item in lr:
            learning_rate_file.write(f"{lr_item} ")

        for epoch_item in epoch:
            epoch_info_file.write(f"{epoch_item} ")

        loss_file.close()
        learning_rate_file.close()
        epoch_info_file.close()

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         LambdaLR(self.num_epoch, self.epoch,
                                                                  self.decay_epoch).step)

        total_step = len(self.train_loader)
        losses = AverageMeter()
        loss_set, lr_set, epoch_set = self.read_loss_info()

        loss_window = self.visdom.line(Y=[1])
        lr_window = self.visdom.line(Y=[1])

        for epoch in range(self.epoch, self.num_epoch):
            losses.reset()
            for step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), self.batch_size)

                if step % 10 == 0:
                    print(f'Epoch [{epoch}/{self.num_epoch}], Step [{step}/{total_step}], Loss: {losses.avg:.4f}')
            loss_set += [losses.avg]
            lr_set += [optimizer.param_groups[0]['lr']]
            epoch_set += [epoch]
            loss_window = self.visdom.line(Y=loss_set, X=epoch_set, win=loss_window, update='replace')
            lr_window = self.visdom.line(Y=lr_set, X=epoch_set, win=lr_window, update='replace')

            self.save_loss_info(loss_set, lr_set, epoch_set)
            torch.save(self.net.state_dict(), '%s/vgg16-%d.pth' % (self.checkpoint_dir, epoch))
            lr_scheduler.step()
