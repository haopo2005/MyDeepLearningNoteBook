import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import RB_path
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    evaluate = None  # evaluate model path
    train_dataset_path = os.path.join(RB_path, 'train')
    val_dataset_path = os.path.join(RB_path, 'valid')

    network = "resnet50"
    pretrained = True
    num_classes = 4
    seed = 0
    input_image_size = 224
    scale = 256 / 224

    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(int(input_image_size * scale)),
            transforms.RandomResizedCrop(input_image_size),
            transforms.ColorJitter(brightness=0.5,
                                   contrast=0.5,
                                   saturation=0.5,
                                   hue=0.5)
        ]))
    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size*scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor()
        ]))

    milestones = [30, 60, 90, 120, 150, 180]
    epochs = 250
    per_node_batch_size = 128  # total batchsize=per_node_batch_size*node_num
    accumulation_steps = 1
    lr = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 8
    print_interval = 100
    apex = True
    sync_bn = True
