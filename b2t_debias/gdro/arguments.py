import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='temp', type=str)

# dataset configuration
parser.add_argument('--dataset', default='celeba', type=str, help='dataset celeba[default]')
parser.add_argument('--data_root', default='/data', type=str, help='dataset root dir')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--target_attr', default=9, type=int)
parser.add_argument('--bias_attr', default=20, type=int)
parser.add_argument('--pseudo_bias', default=None, type=str)
parser.add_argument('--num_classes', default=2, type=int)

# model configuration
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--pretrained', default='imagenet', type=str)

# optimization configuration
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')

parser.add_argument('--seed', type=int, default=1)


def get_arguments():
    args = parser.parse_args()
    return args
