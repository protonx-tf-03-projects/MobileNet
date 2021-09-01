import os
from argparse import ArgumentParser
import tensorflow as tf
from model import MobileNetV1
import numpy as np


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--model-path", default='./model/model.h5', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--rho", help='modify img_size, should be (0:1]',default=1.0, type=float)
    
    home_dir = os.getcwd()
    args = parser.parse_args()
    rho = args.rho
    image_size = args.image_size

    # FIXME
    # Project Description
    print('---------------------Welcome to ProtonX MobileNet-------------------')
    print('Github: https://github.com/protonx-tf-03-projects/MobileNet')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Testing MobileNet model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')


    # Loading Model
    mobilenet = tf.keras.models.load_model(args.model_path) 

    # Load test images from folder
    image = tf.keras.preprocessing.image.load_img(args.test_file_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    x = tf.image.resize(
        input_arr, [(int(image_size*rho)), (int(image_size*rho))]
    )
    x = x / 255

    predictions = mobilenet.predict(x)   
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(predictions))
    print('This image belongs to class: {}'.format(np.argmax(predictions), axis=1))

