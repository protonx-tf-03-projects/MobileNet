import os
from argparse import ArgumentParser

#import tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    # parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--train-folder", default='./data/train', type=str)
    parser.add_argument("--valid-folder", default='./data/validation', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=2, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ProtonX MobileNet-------------------')
    print('Github: https://github.com/NKNK-vn')
    print('Email: khoi.nkn12@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training MobileNet model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    train_folder = args.train_folder
    valid_folder = args.valid_folder
    batch_size =  args.batch_size
    image_size = args.image_size

    # Load train images from folder
    train_ds = image_dataset_from_directory(
        train_folder,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=100
    )

    # Load valid images from folder
    val_ds = image_dataset_from_directory(
        valid_folder,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=100
    )

    assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'

    # FIXME
    # Do Prediction


