# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:41:02 2020

@author: Casper
"""

# import specific
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

def create_model_Cutie(output_classes):
    inputs = layers.Input(shape=(100, 70))
    x = layers.Embedding(35000, 128)(inputs) 
    
    xc = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x = tfa.layers.InstanceNormalization()(xc)
    x = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x2 = tfa.layers.InstanceNormalization()(x2)
    
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    c = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    c = tfa.layers.InstanceNormalization()(c)
    
    aspp1 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(4,4), padding="same", activation = 'relu')(c)
    aspp1 = tfa.layers.InstanceNormalization()(aspp1)
    aspp2 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(8,8), padding="same", activation = 'relu')(c)
    aspp2 = tfa.layers.InstanceNormalization()(aspp2)
    aspp3 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(16,16), padding="same", activation = 'relu')(c)
    aspp3 = tfa.layers.InstanceNormalization()(aspp3)
    
    conc = layers.Concatenate()([aspp1, aspp2, aspp3])
    x = layers.Conv2D(256, (1,1), strides=(1, 1), padding="same", activation = 'relu')(conc)
    x = tfa.layers.InstanceNormalization()(x)
    
    conc = layers.Concatenate()([x, xc])
    x = layers.Conv2D(64, (1,1), strides=(1, 1), padding="same", activation = 'relu')(conc)
    x = tfa.layers.InstanceNormalization()(x)
    
    outputs = layers.Conv2D(output_classes, (1,1), strides=(1, 1), padding="same", activation = 'softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CUTIE_Model")
    return model

def create_model_Bert(output_classes):
    inputs = layers.Input(shape=(100, 70, 768))
    
    xc = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(inputs)
    x = tfa.layers.InstanceNormalization()(xc)
    x = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x = tfa.layers.InstanceNormalization()(x)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), padding="same", activation = 'relu')(x)
    x2 = tfa.layers.InstanceNormalization()(x2)
    
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    x2 = tfa.layers.InstanceNormalization()(x2)
    c = layers.Conv2D(256, (3,5), strides=(1, 1), dilation_rate=(2,2), padding="same", activation = 'relu')(x2)
    c = tfa.layers.InstanceNormalization()(c)
    
    aspp1 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(4,4), padding="same", activation = 'relu')(c)
    aspp1 = tfa.layers.InstanceNormalization()(aspp1)
    aspp2 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(8,8), padding="same", activation = 'relu')(c)
    aspp2 = tfa.layers.InstanceNormalization()(aspp2)
    aspp3 = layers.Conv2D(128, (3,5), strides=(1, 1), dilation_rate=(16,16), padding="same", activation = 'relu')(c)
    aspp3 = tfa.layers.InstanceNormalization()(aspp3)
    
    conc = layers.Concatenate()([aspp1, aspp2, aspp3])
    x = layers.Conv2D(256, (1,1), strides=(1, 1), padding="same", activation = 'relu')(conc)
    x = tfa.layers.InstanceNormalization()(x)
    
    conc = layers.Concatenate()([x, xc])
    x = layers.Conv2D(64, (1,1), strides=(1, 1), padding="same", activation = 'relu')(conc)
    x = tfa.layers.InstanceNormalization()(x)
    
    outputs = layers.Conv2D(output_classes, (1,1), strides=(1, 1), padding="same", activation = 'softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CUTIE_Model")
    return model


