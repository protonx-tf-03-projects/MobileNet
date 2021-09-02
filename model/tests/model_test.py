import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--model-path", default='../model.h5', type=str)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--image-channels", default=3, type=int)
    parser.add_argument("--rho", help='modify img_size, should be (0:1]',default=1.0, type=float)
    
    args = parser.parse_args()
    rho = args.rho

    # FIXME
    # Project Description
    print('---------------------Welcome to ProtonX MobileNet-------------------')
    print('Github: https://github.com/protonx-tf-03-projects/MobileNet')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Testing MobileNet model with hyper-params:')
    print('===========================')
    print(f'Model path: {args.model_path}')
    print(f'Test File Path: {args.test_file_path}')
    print(f'Image size: ({args.image_size}, {args.image_size})')
    
    print('===========================')
 
    # Preprocessing the data:
    test_datagen = ImageDataGenerator(rescale=1./255)
    # Load test dataset:
    test_ds = test_datagen.flow_from_directory(
        args.test_file_path,
        target_size=(int(image_size*rho), int(image_size*rho)),
        batch_size=batch_size,
        class_mode='categorical',
        seed=123
    )
    _, label = test_ds.next()
    mobilenet = tf.keras.models.load_model(args.model_path) 
    probs_predictions = mobilenet.predict(test_ds, verbose=0)
    classes_predictions = mobilenet.predict_classes(test_ds, verbose=0)
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(probs_predictions))
    print('This image belongs to class: {}'.format(classes_predictions), axis=1))
    probs_predictions = probs_predictions[:, 0]
    classes_predictions = classes_predictions[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(label, classes_predictions)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(label, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(label, classes_predictions )
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(label, classes_predictions)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(label, classes_predictions)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(label, probs_predictions)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(label, classes_predictions)
    print('Confusion Matrix')
    print(matrix)


