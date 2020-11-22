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
    evaluate = './checkpoints_bak/latest.pth'  # evaluate model path
    train_dataset_path = os.path.join(RB_path, 'train')
    val_dataset_path = os.path.join(RB_path, 'valid')

    num_classes = 4
    seed = 0
    input_image_size = 224
    scale = 256 / 224

    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(input_image_size),
            transforms.ColorJitter(brightness=0.5,
                                   contrast=0.5,
                                   saturation=0.5,
                                   hue=0.5),
            transforms.ToTensor()
        ]))
    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor()
        ]))

    epochs = 100
    warm_up_epochs = 4
    batch_size = 64
    accumulation_steps = 1
    lr = 0.2
    weight_decay = 5e-5
    momentum = 0.9
    num_workers = 8
    print_interval = 100
    apex = True
