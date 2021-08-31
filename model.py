from threading import main_thread
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, GlobalAveragePooling2D, ZeroPadding2D

class MobileNetV1:
    def __init__(self, img_size, num_classes = 2, alpha = 1.0, rho = 1.0):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = Sequential()
        self.alpha = alpha
        self.rho = rho

    def Standard_Conv(self, filter, stride, padding='same'):
        filter = int(filter * self.alpha)
        self.model.add(Conv2D(filters=filter, kernel_size=3, strides=stride, padding=padding,input_shape=[int(self.img_size * self.rho), int(self.img_size * self.rho), 3]))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        return self.model

    def Depthwise_Layer(self, stride, padding='same'):
        self.model.add(DepthwiseConv2D(kernel_size=3, strides=stride, padding=padding))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        return self.model

    def Pointwise_Layer(self, filter, stride):
        filter = int(filter * self.alpha)
        self.model.add(Conv2D(filters=filter, kernel_size=1, strides=stride))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        return self.model

    def build_model(self):
        #Block 1
        self.model = self.Standard_Conv(32, 2)
        self.model = self.Depthwise_Layer(1)
        self.model = self.Pointwise_Layer(64, 1)
        #Block 2
        self.model = self.Depthwise_Layer(2)
        self.model = self.Pointwise_Layer(128, 1)
        #Block 3
        self.model = self.Depthwise_Layer(1)
        self.model = self.Pointwise_Layer(128, 1)
        #Block 4
        self.model = self.Depthwise_Layer(2)
        self.model = self.Pointwise_Layer(256, 1)
        #Block 5
        self.model = self.Depthwise_Layer(1)
        self.model = self.Pointwise_Layer(256, 1)
        #Block 6
        self.model = self.Depthwise_Layer(2)
        self.model = self.Pointwise_Layer(512, 1)
        #Block 7 >> 11
        for i in range(5):
            self.model = self.Depthwise_Layer(1)
            self.model = self.Pointwise_Layer(512, 1)
        #Block 12
        self.model = self.Depthwise_Layer(2)
        self.model = self.Pointwise_Layer(1024, 1)
        #Block 13
        self.model.add(ZeroPadding2D(padding=4))
        self.model = self.Depthwise_Layer(2, padding='valid')
        self.model = self.Pointwise_Layer(1024, 1) 
        #Fully Connected
        self.model.add(GlobalAveragePooling2D())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=self.num_classes, activation='softmax'))
        return self.model

if __name__ == '__main__':
    model = MobileNetV1(224, 2, 1, 1)
    print(model.build_model().summary())