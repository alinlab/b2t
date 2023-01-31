import argparse
import os
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from resnet import get_model
from data_loader import prepare_data
from arguments import get_arguments

torch.multiprocessing.set_sharing_strategy('file_system')

args = get_arguments()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

print(args)


def build_model():
    model = get_model(args)
        
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class GroupEMA:
    def __init__(self, size, step_size=0.01):
        self.step_size = step_size
        self.group_weights = torch.ones(size).cuda() / size

    def update(self, group_loss, group_count):
        self.group_weights = self.group_weights * torch.exp(self.step_size * group_loss.data)
        self.group_weights = self.group_weights / self.group_weights.sum()
        
        weighted_loss = group_loss @ self.group_weights
        return weighted_loss


def test(model, test_loader, writer, epoch, log='valid'):
    model.eval()
    
    ys = []
    bs = []
    test_losses = []
    corrects = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch['x'].to(device), batch['y'].to(device)
            y_hat = model(inputs)
            test_loss = F.cross_entropy(y_hat, targets, reduction='none')
            _, predicted = y_hat.cpu().max(1)
            correct = predicted.eq(batch['y'])
            
            test_losses.append(test_loss.cpu())
            corrects.append(correct)
            ys.append(batch['y'])
            bs.append(batch['a'])
            
    test_losses = torch.cat(test_losses)
    corrects = torch.cat(corrects)
    ys = torch.cat(ys)
    bs = torch.cat(bs)
    
    num_groups = 4
    group = ys*2 + bs
    group_indices = dict()
    for i in range(num_groups):
        group_indices[i] = np.where(group == i)[0]
    
    print('')
    worst_accuracy = 100
    for i in range(num_groups):
        loss = test_losses[group_indices[i]].mean().item()
        correct = corrects[group_indices[i]].sum().item()
        accuracy = 100. * correct / len(group_indices[i])
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_loss = loss
            worst_correct = correct
            worst_len = len(group_indices[i])
        
        writer.add_scalar(f'{log}/accuracy_group{i}', accuracy, epoch)
        print(f'{log} set - group {i}: Average loss: {loss:.4f}, Accuracy: {correct}/{len(group_indices[i])}({accuracy:.4f}%)')

    writer.add_scalar(f'{log}/accuracy_worst_group', worst_accuracy, epoch)
    print(f'{log} set - worst group: Average loss: {worst_loss:.4f}, Accuracy: {worst_correct}/{worst_len}({worst_accuracy:.4f}%)\n')
    
    loss = test_losses.mean().item()
    correct = corrects.sum().item()
    accuracy = 100. * corrects.sum().item() / len(test_loader.dataset)
    writer.add_scalar(f'{log}/accuracy_average', accuracy, epoch)
    print(f'{log} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)\n')

    return worst_accuracy, accuracy


def train(train_loader, model, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    
    train_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='none')
    num_groups = 4

    for batch_idx, batch in enumerate(train_loader):
        model.train()
        inputs, targets, biases = batch['x'].to(device), batch['y'].to(device), batch['a'].to(device)

        y_hat = model(inputs)
        cost_y = criterion(y_hat, targets)
        prec_train = accuracy(y_hat.data, targets.data, topk=(1,))[0]

        group_idx = targets*2 + biases
        group_map = (group_idx == torch.arange(num_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ cost_y.view(-1)) / group_denom
        
        weighted_loss = group_weight_ema.update(group_loss, group_count)
        loss = weighted_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += cost_y.mean().item()

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Prec@1 %.2f\t' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      prec_train)
                  )
                
    return train_loss/(batch_idx+1)


train_loader, _, valid_loader, test_loader = prepare_data(args)
# create model
model = build_model()

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
else:
    raise NotImplementedError

num_groups = 4
group_weight_ema = GroupEMA(size=num_groups, step_size=0.01)

log_dir = os.path.join('results', args.dataset, args.name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
writer = SummaryWriter(log_dir)

def main():
    best_val_acc, best_val_avg_acc = 0, 0
    best_test_acc, best_test_avg_acc = 0, 0
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer, epoch)
        writer.add_scalar(f'train/train_loss', train_loss, epoch)

        valid_acc, valid_avg_acc = test(model, valid_loader, writer, epoch, 'valid')
        test_acc, test_avg_acc = test(model, test_loader, writer, epoch, 'test')
        
        if valid_acc >= best_val_acc:
            best_val_acc, best_val_avg_acc = valid_acc, valid_avg_acc
            best_test_acc, best_test_avg_acc = test_acc, test_avg_acc
            best_epoch = epoch
            state_dict = {'model': model.state_dict(), 'group_weights': group_weight_ema.group_weights}
            torch.save(state_dict, os.path.join(log_dir, f'epoch_{epoch+1}.pth'))

    print(f'Best worst group accuracy (val) at epoch {best_epoch}: {best_val_acc}')
    print(f'Best average accuracy (val) at epoch {best_epoch}: {best_val_avg_acc}')
    print(f'Best worst group accuracy (test) at epoch {best_epoch}: {best_test_acc}')
    print(f'Best average accuracy (test) at epoch {best_epoch}: {best_test_avg_acc}')


if __name__ == '__main__':
    main()
