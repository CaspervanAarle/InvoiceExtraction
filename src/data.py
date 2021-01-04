# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:42:34 2020

@author: Casper
"""
import os
import random
import numpy as np
import tensorflow as tf
from custom_tokenizer import CustomEmbedder

class Data():
    """ 
    Data object mainly used to generate generators for the input of our neural network
    Args:
        directoryname:  decide which data to use: {SROIE_Exact, SROIE_Fuzzy, CUSTOM_Exact, CUSTOM_Fuzzy, CUSTOM_Exact_filtered}
        seed:           to keep train-test splits constant when recreating this Data object
        exclude_list:   decide if and which templates to exclude in the training set. example= [t1, t2]
    """
    def __init__(self, directoryname, seed, exclude_list=[]):
        if(directoryname.startswith("SROIE")):
            self.CLASSES = 5
        else:
            self.CLASSES = 11
            
        self.SEED = seed
        self.TRAINTESTSPLIT = 0.9
        self.PATH = os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\{}\\".format(directoryname)
        
        input_paths = [x for x in os.listdir(self.PATH) if x.endswith('_input')]
        output_paths = [x for x in os.listdir(self.PATH) if x.endswith('_output')]
        input_paths.sort()
        output_paths.sort()
        self.c = list(zip(input_paths, output_paths))
        self.embedder = CustomEmbedder()
        
        #  and not directoryname == "CUSTOM_Exact_filtered" ???
        if exclude_list == []:
            self._test_random()
        else:
            self._test_unseen(exclude_list)
            
        print("training set length: {}".format(len(self.train)))
        print("test set length: {}".format(len(self.test)))
        
        
    # only use for CUSTOM dataset splitting
    def _test_unseen(self, exclude_list):
        """ this function makes train test split: excludes some templates from test set """
        exclude_list2 = [x + '_' for x in exclude_list]
        self.train = [x for x in self.c if not x[0].startswith(tuple(exclude_list2))]
        random.Random(self.SEED).shuffle(self.train)
        self.test = [x for x in self.c if x[0].startswith(tuple(exclude_list2))]
        random.Random(self.SEED+1).shuffle(self.test)
        
    def _test_random(self):
        """ this function makes train test split based on random seed"""
        random.Random(self.SEED).shuffle(self.c)
        self.train = self.c[:int(self.TRAINTESTSPLIT*len(self.c))]
        self.test = self.c[int(self.TRAINTESTSPLIT*len(self.c)):]
        
    def _train_data_generator_fast(self):
        """ endless train data generator from preprocessed data """
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
                        x_ = self.embedder.get_embedding_grid_from_word(x)
                        yield (x_, y)
                        
    # seen templates:
    def _validation_data_generator_fast(self):
        """ endless validation data generator from preprocessed data """
        input_paths, output_paths = zip(*self.test)
        for inputs, outputs in zip(input_paths[:70], output_paths[:70]):
            with open(self.PATH + inputs, 'rb') as i:
                with open(self.PATH + outputs, 'rb') as o:
                    x = np.load(i)
                    y = np.load(o)
                    i.close()
                    o.close()
                    x_ = self.embedder.get_embedding_grid_from_word(x)
                    yield (x_, y)
                
                
    def get_test_list(self):
        return self.test
    
    def get_train_list(self):
        return self.train
                
    def get_train_dataset(self):
        return tf.data.Dataset.from_generator(self._train_data_generator_fast, output_shapes = ((100,70, 768), (100,70, self.CLASSES)), output_types = (tf.float32, tf.float32))
    
    def get_validation_dataset(self):
        return tf.data.Dataset.from_generator(self._validation_data_generator_fast, output_shapes = ((100,70, 768), (100,70, self.CLASSES)), output_types = (tf.float32, tf.float32))
    
    
    
    
    
    
