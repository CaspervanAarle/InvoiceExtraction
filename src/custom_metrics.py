# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:08:45 2020

Metric functions to be applied when training or testing a neural network.

@author: Casper van Aarle
"""
import numpy as np
from fuzzywuzzy import fuzz


def strict_average_precision(y_true, y_pred):
    """ computes a decimal fraction of the predictions that were actually in the answer"""
    count = 0
    correct = 0
    for y_t, y_p in zip(y_true, y_pred):
        for i in range(len(y_t[0,0,:])):
            boolean = _is_correct(y_t[:,:,i], y_p[:,:,i])
            if(boolean):
                correct +=1
            count +=1
    return correct/count

def soft_average_precision(y_true, y_pred):
    """ computes a decimal fraction of the positive predictions that were actually in the answer"""
    
    count = 0
    correct = 0
    for y_t, y_p in zip(y_true, y_pred):
        for i in range(len(y_t[0,0,:])):
            boolean = _is_correct_positives(y_t[:,:,i], y_p[:,:,i])
            if(boolean):
                correct +=1
            count +=1
    return correct/count


def discovery_rate(x_words, y_true, json, lbl_list):
    """ returns for every class whether it was  correctly detected in the grid """
    dictio = {}
    for i in range(len(y_true[0,0,:])):
        label_values = json[lbl_list[i]].split()
        found_words = np.where(y_true[:,:,i], x_words, '').flatten()
        discovered = all([_check_word_in_text(word, found_words) for word in label_values])
        dictio[lbl_list[i]] = discovered
    return dictio
                

# helper functions:

def _check_word_in_text(word, text_list):
    """ checks if word exists in a list with an error margin """
    boolean = any([fuzz.ratio(text_word, word) > 80 for text_word in text_list])
    return boolean
            
def _is_correct(true_values, predict_values):
    """ checks if binary arrays are identical, after rounding """
    threshold = 0.5
    predict_values = np.where(predict_values > threshold, 1, 0)
    return np.array_equal(predict_values, true_values)

def _is_correct_positives(true_values, predict_values):
    """ checks if the positive cells of the prediction are also positive in the true values."""
    threshold = 0.5
    predict_values = np.where(predict_values > threshold, 1, 0)
    new = np.where(true_values > threshold, predict_values, true_values)
    return np.array_equal(new, true_values)



        
if __name__ == '__main__':
    x_words = np.array([["search", "for"],["the", "words"]])
    y_true = np.array([[[1,0], [1,0]],[[0,1], [0,1]]])
    json = {"key1":"search arbitrary", "key2":"the wordsa" }
    discovery_rate(x_words, y_true, json)
    
    
    
    
    
    