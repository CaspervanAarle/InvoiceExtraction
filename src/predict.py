# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:21:53 2020

@author: Casper
"""

import numpy as np
import sys
import os
import pytesseract
import tensorflow as tf
import model
from preprocessing import preprocess_single_input
import json
from data import Data
from custom_tokenizer import CustomEmbedder
import custom_metrics
from fuzzywuzzy import fuzz
from collections import Counter 
import settings
import argparse
from datetime import datetime

# class used to test single predictions]
class SinglePredictor():
    
    def __init__(self, saved_weights_location):
        self.embedder = CustomEmbedder()
        try:
            self.mymodel = model.create_model_Bert(5)
            self._load_my_weights(saved_weights_location)
            self.label_names = ['company', 'date', 'address', 'total']
        except:
            try:
                self.mymodel = model.create_model_Bert(11)
                self._load_my_weights(saved_weights_location)
                self.label_names = ['Factuurdatum', 'Factuurnummer', 'Uitschrijver', 'Bedrag', 'Item1-naam', 'Item1-prijs', 'Item2-naam', 'Item2-prijs', 'Item3-naam', 'Item3-prijs']
            except:
                print("weights do not fit any model or weight directory does not exist")
            
            
    def predict_from_path(self, invoice_path):
        word_grid = preprocess_single_input(invoice_path)
        print(type(word_grid))
        embedding = self.embedder.get_embedding_grid_from_word(np.asarray(word_grid))
        y = self.mymodel(np.expand_dims(embedding, 0))[0]
        
        print(y.shape)
    
            
    def activation_to_prediction(self, invoice_path):
        self.stringbuild = ""
        self.stringbuild += "predict:\n"
        
        word_grid = preprocess_single_input(invoice_path)
        embedding_input = self.embedder.get_embedding_grid_from_word(np.asarray(word_grid))
        out = self.mymodel.predict(np.expand_dims([embedding_input], -1))[0]
        out = np.rollaxis(out, -1, 0)
        for i, channel in enumerate(out[:-1]):
            # do prediction:
            self.stringbuild += self.label_names[i] + ":\n" 
            out_words = np.where(channel > 0.5, word_grid, '').flatten()
            out_words = ' '.join(out_words).split()
            self.stringbuild += str(out_words) + "\n"
        print(self.stringbuild)
        
        
    def _load_my_weights(self, saved_weights_directory): 
        print(saved_weights_directory)
        self.mymodel.load_weights(tf.train.latest_checkpoint("model_checkpoints\\" + saved_weights_directory))

            


# class used to make predictions over the test set
class TestPredictor():
    """ 
    Predictor object
    params:
        saved_weights_location: the directory of the weights
        directory: which directory to do a prediction on
        seed: for Data object to use the same train-test split
        exclude_list (optional): for the CUSTOM dataset, to see performance on unseen templates
    
    """
    def __init__(self, saved_weights_directory, directory, seed, exclude_list=[]):
        self.data = Data(directory, seed, exclude_list)
        self.directory = directory
        self.input_directory = os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\{}\\".format(directory)
        self.threshold = 0.5
        self.embedder = CustomEmbedder()
        
        if(directory.startswith("SROIE")):
            self.mymodel = model.create_model_Bert(5)
            self.label_names = ['company', 'date', 'address', 'total']
            self.labels_directory = os.path.dirname(os.getcwd()) + "\\data\\raw\\SROIE\\"
            self.softAP = {'company':0, 'date':0, 'address':0, 'total':0}
            self.strictAP = {'company':0, 'date':0, 'address':0, 'total':0}
            
        if(directory.startswith("CUSTOM")):
            self.mymodel = model.create_model_Bert(11)
            self.label_names = ['Factuurdatum', 'Factuurnummer', 'Uitschrijver', 'Bedrag', 'Item1-naam', 'Item1-prijs', 'Item2-naam', 'Item2-prijs', 'Item3-naam', 'Item3-prijs']
            self.labels_directory = os.path.dirname(os.getcwd()) + "\\data\\raw\\CUSTOM\\"
            self.softAP = {'Factuurdatum':0, 'Factuurnummer':0, 'Uitschrijver':0, 'Bedrag':0, 'Item1-naam':0, 'Item1-prijs':0, 'Item2-naam':0, 'Item2-prijs':0, 'Item3-naam':0, 'Item3-prijs':0}
            self.strictAP = {'Factuurdatum':0, 'Factuurnummer':0, 'Uitschrijver':0, 'Bedrag':0, 'Item1-naam':0, 'Item1-prijs':0, 'Item2-naam':0, 'Item2-prijs':0, 'Item3-naam':0, 'Item3-prijs':0}
        
        self._load_my_weights(saved_weights_directory)
        
        self.stringbuild = ""
        self.prediction_output_path = "C:\\Users\\Casper\Projects\\Topicus\\implementations\\prj_final\\predictions\\{}_{}.txt".format(directory,datetime.now())
        
       
        
    def predict_test_set(self):
        test_set = self.data.get_test_list()
        for file in test_set:
            self._predict_img_to_string(self.input_directory + file[0])
            self._print_labels(self.labels_directory + file[0][:-6] +"_json.json")
        self._save_output(self.prediction_output_path)
        
    def precision_test_set(self):
        test_set = self.data.get_test_list()   
        for t in range(1): 
            for file in test_set:
                if(self.directory.startswith("CUSTOM")):
                    self._get_precision(self.input_directory + file[0], self.labels_directory + file[0][:-6] + "_json.json")
                else:
                    self._get_precision(self.input_directory + file[0], self.labels_directory + file[0][:-6] + ".txt")
                    
            print(self.softAP)
            print(self.strictAP)
            print(sum(self.strictAP.values()))
    
    def check_grid_labels(self):
        test_set = self.data.get_test_list()
        for file in test_set:
            self._check_grid_labels(self.input_directory + file[0], self.labels_directory + file[0][:-6] +"_json.json")
    
    
    def _get_detection_rate(self, x_words, y_true, json, json_lbls):
        detection = custom_metrics.discovery_rate(x_words, y_true, json, json_lbls)
        self.detectionrate = dict(Counter(self.detectionrate), Counter(detection))
        
        
        
    def _predict_img_to_string(self, input_path):
        self.stringbuild += "------------------\n"
        self.stringbuild += "predict:\n"
        with open(input_path, 'rb') as i:
            word_input = np.load(i)
            i.close()
        embedding_input = self.embedder.get_embedding_grid_from_word(word_input)
        out = self.mymodel.predict(np.expand_dims([embedding_input], -1))[0]
        out = np.rollaxis(out, -1, 0)
        for i, channel in enumerate(out[:-1]):
            # do prediction:
            self.stringbuild += self.label_names[i] + ":\n" 
            out_words = np.where(channel > self.threshold, word_input, '').flatten()
            out_words = ' '.join(out_words).split()
            self.stringbuild += str(out_words) + "\n"
            
            
            
            
    # string outputs        
    def _get_precision(self, input_path, json_path): 
        # open json output
        json_label = self._load_json(json_path)
        
        # open input
        with open(input_path, 'rb') as i:
            word_input = np.load(i)
            i.close()
            
        with open(input_path[:-6] + "_output", 'rb') as i:
            output = np.load(i)
            i.close()
            
        # generate output
        embedding_input = self.embedder.get_embedding_grid_from_word(word_input)
        out = self.mymodel.predict(np.expand_dims([embedding_input], -1))[0]
        out = np.rollaxis(out, -1, 0)
        out_real = np.rollaxis(output, -1, 0)
        
        # predict for every channel
        #print("------------")
        #print("ground truth: ")
        for i, channel in enumerate(out[:-1]):
            # do prediction:
            out_words = np.where(channel > self.threshold, word_input, '').flatten()
            out_words = ' '.join(out_words).split()
            out_words = list(set(out_words))
            label_words = str(json_label[self.label_names[i]]).split()
            label_words = [x for x in label_words if x != 'nan']
            #print(label_words)
            result_softap =  all(self._elem_in_list(elem, out_words)  for elem in label_words)
            self.softAP[self.label_names[i]] += result_softap
            
            part_result = all(self._elem_in_list(elem, label_words)  for elem in out_words)
            result_strictap = result_softap and part_result
            self.strictAP[self.label_names[i]] += result_strictap
        
        #print("grid labels: ")
        for i, channel in enumerate(out_real[:-1]):
            out_words = np.where(channel > self.threshold, word_input, '').flatten()            
            out_words = ' '.join(out_words).split()
            out_words = list(set(out_words))
            #print(out_words)
            
    # string outputs            
    def _check_grid_labels(self, input_path, json_path):
        #open json output
        json_label = self._load_json(json_path)
        
        # open grid output
        with open(input_path[:-6] + "_output", 'rb') as i:
            output = np.load(i)
            i.close()
            
        # open word input
        with open(input_path, 'rb') as i:
            word_input = np.load(i)
            i.close()
            
        # generate output
        out_real = np.rollaxis(output, -1, 0)
    
        print("------------")
        print("ground truth: ")
        for i, channel in enumerate(out_real[:-1]):
            label_words = str(json_label[self.label_names[i]]).split()
            label_words = [x for x in label_words if x != 'nan']
            print(label_words)
        
        print("grid labels: ")
        for i, channel in enumerate(out_real[:-1]):
            out_words = np.where(channel > self.threshold, word_input, '').flatten()            
            out_words = ' '.join(out_words).split()
            out_words = list(set(out_words))
            print(out_words)
            
            
    def _elem_in_list(self, elem, alist):
        for out in alist:
            if(fuzz.ratio(elem, out) > 80):
                return True
        return False
            
    
    def _print_labels(self, input_path):
        self.stringbuild += "true:\n"
        json_label = self._load_json(input_path)
        self.stringbuild += json.dumps(json_label) + "\n"
        
            
    def _load_json(self, path):
        try:
            string = open(path, 'r').read()
        except:
            print("json not found")
            pass
        return json.loads(string)


    def _save_output(self, path):
        file = open(path, "wb") 
        file.write(self.stringbuild.encode('utf-8'))
        file.close() 
    
    def _load_my_weights(self, saved_weights_directory): 
        if(saved_weights_directory == ""):
            saved_weights_directory = self.directory + "\\" + os.listdir(os.getcwd() + "\\model_checkpoints\\" + self.directory + "\\")[-1]
        print(saved_weights_directory)
        self.mymodel.load_weights(tf.train.latest_checkpoint("model_checkpoints\\" + saved_weights_directory))

   
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('input_dir', help='which preprocessed data directory to use (name of the directory in the data/preprocessed folder')
    parser.add_argument('input', help='invoice or preprocessed invoice directory ')
    
    parser.add_argument('-e', '--exclude_items', nargs="+", default=[], help='which templates should be excluded when using CUSTOM dataset')
    parser.add_argument('-cp', '--checkpoint', default="", help='directory of the checkpoint to use in the model_checkpoints folder for the prediction')
    parser.add_argument('-s', '--splitseed', default=14202, help='seed to split data into train-test groups')
    args = parser.parse_args()
    
    if(args.input in os.listdir(os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\")):
        predictor = TestPredictor(args.checkpoint, args.input, args.splitseed, args.exclude_items)
        predictor.precision_test_set()
    if(os.path.isfile(args.input)):
        assert(args.checkpoint)
        predictor = SinglePredictor(args.checkpoint)
        predictor.activation_to_prediction(args.input)
        
        
        
        
    
    
        
