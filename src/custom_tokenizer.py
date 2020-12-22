# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:53:15 2020

@author: Casper
"""
from transformers import BertTokenizer, TFBertModel
from transformers import BertModel
import tensorflow as tf
import numpy as np
from time import time 
import sys


class CustomTokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bertmodel = TFBertModel.from_pretrained('bert-base-uncased')
        
            
    # CUTIE method (output = index)
    def get_index_grid_from_word(self, grid):
        shape = np.array(grid).shape
        output = np.zeros((shape[0],shape[1]))
        for i, ilist in enumerate(grid):
            for j, word in enumerate(ilist):
                output[i][j]=np.max(self._word_to_int(word))
        return(output)
    
    
    def _word_to_int(self, word):
        return self.tokenizer.encode(word)
    
    
    def int_to_word(self, integ):
        return self.tokenizer.decode(integ)
    
    
    
    
class CustomEmbedder():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bertmodel = TFBertModel.from_pretrained('bert-base-uncased')
        np.set_printoptions(threshold=sys.maxsize)
    
    
    # BERTgrid method (output = embedding)
    def get_embedding_grid_from_word(self, grid):  
        grid_embedding = np.zeros((7000,768))     
        
        # get all words:
        text_array = grid.flatten().tolist() # make list
        text_array = [x if not x.isspace() else '' for x in text_array] # remove spaces
        text_array_input = [x for x in text_array if x] #remove empty slots
        
        # base case (no data available):
        if (len(text_array_input) == 0):
            return np.reshape(grid_embedding, (100,70,768))
        # generate tokens:
        inputs = self.tokenizer.batch_encode_plus(text_array_input, return_tensors="np", add_special_tokens = False, truncation = True, padding = True)
        
        # extra preprocessing:
        red1 = np.expand_dims(np.amax(inputs['input_ids'], axis=-1), 0)
        red2 = np.expand_dims(np.amax(inputs['token_type_ids'], axis=-1), 0)
        red3 = np.expand_dims(np.amax(inputs['attention_mask'], axis=-1), 0)
        new_input = {'input_ids': red1, 'token_type_ids': red2, 'attention_mask': red3}
        
        # get all embeddings:
        output = self.bertmodel(new_input)
        embedding = output[0][0]
        j = 0
        for i, word in enumerate(text_array):
            if len(word) != 0:
                grid_embedding[i] = embedding[j]
                j+=1
        grid_embedding = np.reshape(grid_embedding, (100,70,768))
        return grid_embedding
            
    
    
    
    
    