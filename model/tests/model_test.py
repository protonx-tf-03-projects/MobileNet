import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision

def recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall

def f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1

def matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return matrix

def metric(i):
    switcher={
                1: accuracy(label, classes_predictions),
                2: precision(label, classes_predictions),
                3: recall(label, classes_predictions),
                4: f1(label, classes_predictions),
                5: matrix(label, classes_predictions)
            }
    return switcher.get(i,'Invalid number')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test-folder", type=str, required=True)
    parser.add_argument("--model-path", default='../model.h5', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--rho", help='modify img_size, should be (0:1]',default=1.0, type=float)
    parser.add_argument("--batch-size", default=16, type=int)

    args = parser.parse_args()
    rho = args.rho
    batch_size =  args.batch_size
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

    # Preprocessing the data:
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load test dataset:
    test_ds = test_datagen.flow_from_directory(
        args.test_folder,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    #Get y_true
    label = test_ds.classes

    #Get y_pred
    mobilenet = tf.keras.models.load_model(args.model_path) 
    probs_predictions = mobilenet.predict(test_ds, verbose=0)
    classes_predictions = np.argmax(probs_predictions, axis=1)
    while(True):
        print("\n0: End process")
        print("1: Accuracy")
        print("2: Precision")
        print("3: Recall")
        print("4: F1_score")
        print("5: Confusion matrix")
        print("Please select a metric for evaluating model: ")
        x = input()
        x = int(x)
        if x == 0:
            break
        print('---------------------Prediction Result: -------------------')
        print(metric(x))
