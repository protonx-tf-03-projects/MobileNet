import os
from argparse import ArgumentParser

#import tensorflow
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.engine.sequential import Sequential
from model import MobileNetV1
from keras.applications.mobilenet import preprocess_input

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--model-folder", default='./model/', type=str)
    parser.add_argument("--train-folder", default='./data/train', type=str)
    parser.add_argument("--valid-folder", default='./data/validation', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--alpha", help='modify filter, should be (0:1]',default=1.0, type=float)
    parser.add_argument("--rho", help='modify img_size, should be (0:1]',default=1.0, type=float)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ProtonX MobileNet-------------------')
    print('Github: https://github.com/protonx-tf-03-projects/MobileNet')
    print('Email: ') #Update later
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
    alpha = args.alpha
    rho = args.rho
    num_classes = args.num_classes

    #Use ImageDataGenerator for augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                    width_shift_range=0.2,height_shift_range=0.2,
                                    shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    #Load train set
    train_ds = train_datagen.flow_from_directory(
        train_folder,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123
    )
    #Load test set
    val_ds = val_datagen.flow_from_directory(
        valid_folder,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=123
    )

    # assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    assert alpha > 0 and alpha <= 1, 'Unfortunately, model accepts alpha  with lower than 1 and higher than 0'
    assert rho > 0 and rho <= 1, 'Unfortunately, model accepts alpha  with lower than 1 and higher than 0'
    assert image_size > 32, 'Unfortunately, model accepts jpg images size higher than 32'

    #Load model
    MobileNet = MobileNetV1(image_size, num_classes, alpha, rho)
    model = MobileNet.build_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs = args.epochs,validation_data = val_ds)
    # FIXME
    # Do Prediction


