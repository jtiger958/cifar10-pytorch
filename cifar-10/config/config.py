import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--epoch', type=int, default=196, help='start number of epochs to train for')
parser.add_argument('--num_epoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--decay_epoch', type=int, default=40, help='decay epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--loss_dir', default='losses_check', help="path to loss information while training")
parser.add_argument('--num_workers', type=int, default=4, help="")


def get_config():
    return parser.parse_args()
