import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from public.path import RB_path
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config(object):
    log = './log_b3'  # Path to save log
    checkpoint_path = './checkpoints'  # Path to store checkpoint model
    resume = './checkpoints/latest.pth'  # load checkpoint model
    #evaluate = './checkpoints/efficientnet_b3-epoch150-acc96.26373621678614.pth'  # evaluate model path
    evaluate = None
    train_dataset_path = os.path.join(RB_path, 'train')
    val_dataset_path = os.path.join(RB_path, 'valid')

    network = "efficientnet_b3"
    pretrained = False
    num_classes = 4
    seed = 0
    input_image_size =256
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
            transforms.Resize(int(input_image_size*scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor()
        ]))

    warm_up_epochs = 10
    epochs = 300
    batch_size = 8
    accumulation_steps = 1
    lr = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    num_workers = 4
    print_interval = 100
    apex = True
