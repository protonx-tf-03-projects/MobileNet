import os
from argparse import ArgumentParser

#import tensorflow
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax, SGD
from model import MobileNetV1
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=16, type=int)
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
    train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    #Load train set
    train_ds = train_datagen.flow_from_directory(
        train_folder,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=10,
    )
    #Load test set
    val_ds = val_datagen.flow_from_directory(
        valid_folder,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=10,
    )

    # train_ds = image_dataset_from_directory(
    #     train_folder,
    #     seed=123,
    #     image_size=(int(image_size*rho), int(image_size*rho)),
    #     shuffle=True,
    #     batch_size=batch_size, 
    #     label_mode='categorical'
    # )

    # # Load Validation images from folder
    # val_ds = image_dataset_from_directory(
    #     valid_folder,
    #     seed=123,
    #     image_size=(int(image_size*rho), int(image_size*rho)),
    #     shuffle=True,
    #     batch_size= batch_size,
    #     label_mode='categorical'
    # )

    # assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    assert alpha > 0 and alpha <= 1, 'Unfortunately, model accepts alpha  with lower than 1 and higher than 0'
    assert rho > 0 and rho <= 1, 'Unfortunately, model accepts alpha  with lower than 1 and higher than 0'
    assert image_size > 32, 'Unfortunately, model accepts jpg images size higher than 32'

    #Load model
    MobileNet = MobileNetV1(image_size, num_classes, alpha, rho)
    model = MobileNet.build_model()

    #Train model
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, epochs = args.epochs,validation_data = val_ds)

    #Show Model Train History
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model training')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper left')
    plt.savefig("train.jpg")
    plt.show()

    #Show Model Val History
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model validation')
    plt.ylabel('value')
    plt.xlabel('epoch')
    plt.legend(['val_loss', 'val_accuracy'], loc='upper left')
    plt.savefig("val.jpg")
    plt.show()
    # Do Prediction


