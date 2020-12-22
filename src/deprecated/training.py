# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:38:50 2020

@author: Casper
"""

# import base
import os
import random
import numpy as np
from datetime import datetime

# import specific
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa

# import custom
import custom_metrics
import preprocessing


class Data():
    
    def __init__(self):
        self.TRAINTESTSPLIT = 0.8
        self.PATH = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\prj_new_data\\dataset_preprocessed\\"
        input_paths = [x for x in os.listdir(self.PATH) if x.endswith('_input')]
        output_paths = [x for x in os.listdir(self.PATH) if x.endswith('_output')]
        input_paths.sort()
        output_paths.sort()
        c = list(zip(input_paths, output_paths))
        random.Random(14045).shuffle(c)
        self.train = c[:int(self.TRAINTESTSPLIT*len(c))]
        self.test = c[int(self.TRAINTESTSPLIT*len(c)):]
    
    def _train_data_generator_fast(self):
        while(True):
            input_paths, output_paths = zip(*self.train)
            #mg = MutationGenerator().generate_mutation()
            for inputs, outputs in zip(input_paths, output_paths):
                with open(self.PATH + inputs, 'rb') as i:
                    with open(self.PATH + outputs, 'rb') as o:
                        x = np.load(i)
                        y = np.load(o)
                        i.close()
                        o.close()
                        y_ = self._add_dont_care(y)
                        yield (x, y_)
    # seen templates:
    def _validation_data_generator_fast(self):
        input_paths, output_paths = zip(*self.test)
        #mg = MutationGenerator().generate_mutation()
        for inputs, outputs in zip(input_paths, output_paths):
            with open(self.PATH + inputs, 'rb') as i:
                with open(self.PATH + outputs, 'rb') as o:
                    x = np.load(i)
                    y = np.load(o)
                    i.close()
                    o.close()
                    y_ = self._add_dont_care(y)
                    yield (x, y_)
    # unseen templates:
    def _validation_data_generator_fast2(self):
        self.PATH = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\prj_new_data\\dataset_preprocessed\\"
        input_paths, output_paths = zip(*self.test)
        #mg = MutationGenerator().generate_mutation()
        for inputs, outputs in zip(input_paths, output_paths):
            with open(self.PATH + inputs, 'rb') as i:
                with open(self.PATH + outputs, 'rb') as o:
                    x = np.load(i)
                    y = np.load(o)
                    i.close()
                    o.close()
                    y_ = self._add_dont_care(y)
                    yield (x, y_)
                
    def get_train_dataset(self):
        return tf.data.Dataset.from_generator(self._train_data_generator_fast, output_shapes = ((70,50), (70, 50, 11)), output_types = (tf.float32, tf.float32))
    
    def get_validation_dataset(self):
        return tf.data.Dataset.from_generator(self._validation_data_generator_fast, output_shapes = ((70,50), (70, 50, 11)), output_types = (tf.float32, tf.float32))
    
    
    def _add_dont_care(self, array):
        care = 1*(array.sum(axis=2) == 1)
        dontcare = 1 - care
        dontcare = np.expand_dims(dontcare, axis=-1)
        #print(dontcare.shape)
        #print(array.shape)
        return np.concatenate((array, dontcare), axis=-1)
    
def _create_model_Cutie():
    inputs = layers.Input(shape=(70, 50))
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
    
    outputs = layers.Conv2D(11, (1,1), strides=(1, 1), padding="same", activation = 'softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CUTIE_Model")
    model.summary()
    return model

def learn_model(model =_create_model_Cutie()):
    
    # create model and datasets:
    data = Data()
    mydataset = data.get_train_dataset().batch(4)
    myvalidationset = data.get_validation_dataset().batch(4)
    
    # create metrics:
    #met = tf.keras.metrics
    #prec = [met.Recall(thresholds=0.1, class_id=1, name='date_recall'),met.Recall(thresholds=0.1, class_id=3, name='total_recall'), met.Precision(thresholds=0.1, class_id=1, name = 'date_precision'),met.Precision(thresholds=0.1, class_id=3, name='total_precision')]
    
    # create optimizer:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # create callbacks:
    logdir = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\prj_new_data\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = "model_checkpoints/" + datetime.now().strftime("%Y%m%d-%H%M%S") +"/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    #c_o_callback = ConsoleOutputCallback()
    #roccallback = ConsoleROCCallback([1,3])
    
    
    # set checkpoint:
    checkpoint_path = "model_checkpoints/perm_test/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    

    crossent_loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer, loss = crossent_loss, metrics = [custom_metrics.strict_average_precision, custom_metrics.soft_average_precision])
    model.run_eagerly = True
    with tf.device('/gpu:0'):
        model.fit(mydataset, steps_per_epoch = 30, epochs = 1000, batch_size = 8, validation_data = myvalidationset, callbacks=[cp_callback, tensorboard_callback])
        
        
if __name__ == '__main__':
    
    model = _create_model_Cutie()
    model.load_weights(tf.train.latest_checkpoint('model_checkpoints\\20201020-154916\\'))
    #learn_model(model)
    
    input_path = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\prj_new_data\\t1_9_pdf.pdf"
    float_input, word_input = preprocessing.preprocess_single_input(input_path)
    model.predict(float_input)
    