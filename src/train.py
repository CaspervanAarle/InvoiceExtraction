# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:38:50 2020

@author: Casper van Aarle
"""

# import base
from datetime import datetime
import os

# import specific
import tensorflow as tf

# import custom
import custom_metrics
import model
from data import Data
import settings
import argparse


def learn_model(directory, seed, exclude_list=[], saved_weights_location=""):
    # create model:
    if(directory.startswith("SROIE")):
        mymodel = model.create_model_Bert(5)
    else:
        mymodel = model.create_model_Bert(11)
        
    # create dataset:
    data = Data(directory, seed, exclude_list)
    mydataset = data.get_train_dataset().batch(2)
    myvalidationset = data.get_validation_dataset().batch(2)
    
    # re-use weights if provided:
    if(saved_weights_location != ""):
        mymodel.load_weights(tf.train.latest_checkpoint("model_checkpoints\\" + directory + "\\" + saved_weights_location))
        
    # create metrics:
    #met = tf.keras.metrics
    #my_custom_metrics = [met.Recall(thresholds=0.1, class_id=1, name='date_recall'),met.Recall(thresholds=0.1, class_id=3, name='total_recall'), met.Precision(thresholds=0.1, class_id=1, name = 'date_precision'),met.Precision(thresholds=0.1, class_id=3, name='total_precision')]
    my_custom_metrics = [custom_metrics.soft_average_precision, custom_metrics.strict_average_precision]
    
    # create optimizer:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # create tensorboard callback:
    logdir = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\prj_final\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    # create checkpoint callback:
    if not os.path.exists(os.getcwd() + "\\model_checkpoints\\" + directory + "\\"):
        os.makedirs(os.getcwd() + "\\model_checkpoints\\" + directory + "\\")
    checkpoint_path = "model_checkpoints\\" + directory + "\\" + datetime.now().strftime("%Y%m%d-%H%M%S") +"/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    
    # create loss function
    crossent_loss = tf.keras.losses.CategoricalCrossentropy()
    
    mymodel.compile(optimizer = optimizer, loss = crossent_loss, metrics=my_custom_metrics)
    mymodel.run_eagerly = True
    with tf.device('/gpu:0'):
        mymodel.fit(mydataset, steps_per_epoch = 100, epochs = 1000, batch_size = 8, validation_data = myvalidationset, callbacks=[tensorboard_callback, cp_callback])
        
        
        
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        
if __name__ == '__main__':   

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input', help='which preprocess data to use (name of the directory in the data/preprocessed folder')
    parser.add_argument("-e", "--exclude_items", nargs="+", default=[], help='which templates should be excluded when using CUSTOM dataset')
    parser.add_argument('-cp', '--checkpoint', default="", help='directory of the checkpoint to use in the model_checkpoints folder to continue learning')
    parser.add_argument('-s', '--splitseed', default=14202, help='seed to split data into train-test groups')
    args = parser.parse_args()
    
    assert(args.input)
    assert(not (args.input.startswith("SROIE") and len(args.exclude_items)  > 0))
    assert(is_int(args.splitseed))
    
    learn_model(args.input, int(args.splitseed), args.exclude_items, args.checkpoint)
    