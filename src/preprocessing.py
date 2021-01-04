# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:47:50 2020

@author: Casper
"""
import pytesseract
from pytesseract import Output
import os
import pdf2image
import json
import gridgenerator as gg
import numpy as np
import pandas as pd
import random
from fuzzywuzzy import fuzz
import settings

import sys
import argparse


# preprocess all the data in the directory and save as numpy files:
def preprocess_data(dataset):
    """ preprocess one of two datasets (SROIE or CUSTOM)
    
    multiple preprocessing folders are filled with "Exact Matching", "Fuzzy Matching" grids
    the "Exact Filtered" folder is filled and some templates are left out because of wrong formats in the invoices
    
    Args:
        dataset (str) : "SROIE" or "CUSTOM" string
    """
    print("preprocess raw {} dataset".format(dataset))
    GRIDSIZE = [100,70]
    
    GOAL_PATH_Exact = os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\{}_Exact\\".format(dataset)
    GOAL_PATH_Fuzzy = os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\{}_Fuzzy\\".format(dataset)
    
    # SET DATASET:
    data_loader = _load_raw_SROIE
    grid_generator = gg.generate_io_grid_SROIE
    
    if(dataset == "CUSTOM"):
        GOAL_PATH_Exact_Filtered = os.path.dirname(os.getcwd()) + "\\data\\preprocessed\\{}_Exact_Filtered\\".format(dataset)
        data_loader = _load_raw
        grid_generator = gg.generate_io_grid
    
    # GENERATE NUMPY IO DATA:
    for i, (ip, op, name) in enumerate(data_loader()):
        (df, r_error) = _label_dataframe_SCC(ip, op)
        (df2, r_error) = _label_dataframe_Exact(ip, op)
        
        (in_grid, out_grid) = grid_generator(df, GRIDSIZE)
        (in_grid2, out_grid2) = grid_generator(df2, GRIDSIZE)
        
        _save_output(in_grid,  GOAL_PATH_Exact + name + '_input')
        _save_output(out_grid, GOAL_PATH_Exact + name + '_output')
        
        template_filter = ("t17", "t18", "t19", "t20")
        if(dataset == "CUSTOM" and not name.startswith(template_filter)):
            _save_output(in_grid,  GOAL_PATH_Exact_Filtered + name + '_input')
            _save_output(out_grid, GOAL_PATH_Exact_Filtered + name + '_output')
        
        _save_output(in_grid2,  GOAL_PATH_Fuzzy + name + '_input')
        _save_output(out_grid2, GOAL_PATH_Fuzzy + name + '_output')
        
def preprocess_single_input(input_path):
    """ preprocess input from specific path and return the input grid and the corresponding word grid """
    GRIDSIZE = [100,70]
    try:
        img = pdf2image.convert_from_path(input_path)[0]
    except:
        print("no such path")
    df = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
    word_grid = gg.generate_i_grid(df, GRIDSIZE)
    return word_grid


# load raw data to dataframe
def _load_raw():
    ds_path = os.path.dirname(os.getcwd()) + "\\data\\raw\\CUSTOM\\"
    filenames_pdf = [x for x in os.listdir(ds_path) if x.endswith('.pdf')]
    filenames_json = [x for x in os.listdir(ds_path) if x.endswith('.json')]
    c = list(zip(filenames_pdf, filenames_json))
    random.shuffle(c)
    filenames_pdf, filenames_json = zip(*c)
    for fn_pdf, fn_json in zip(filenames_pdf, filenames_json):
        img = pdf2image.convert_from_path(ds_path + fn_pdf)[0]
        df = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
        string = open(ds_path + fn_json, 'r').read()
        dict_v = json.loads(string)
        yield (df, dict_v, fn_pdf[:-8])
        
# load raw data to dataframe
def _load_raw_SROIE():
    ds_path = os.path.dirname(os.getcwd()) + "\\data\\raw\\SROIE\\"
    filenames_jpg = [x for x in os.listdir(ds_path) if x.endswith('.jpg')]
    random.shuffle(filenames_jpg)
    for fn_jpg in filenames_jpg:
        df = pytesseract.image_to_data(ds_path + fn_jpg, output_type=Output.DATAFRAME)
        dict_v = load_json(ds_path + fn_jpg[:-4] + ".txt")
        yield (df, dict_v, fn_jpg[:-4])

   
# add labels to df with exact matching
def _label_dataframe_Exact(df, json):
    not_found = 0
    labels = json.keys()
    for lbl in labels:
        # exclude labels:
        if lbl.endswith('-hoeveel'):
            continue
        if lbl == 'Template':
            continue
        if json[lbl] != json[lbl]:
            #print('label unavailable')
            continue
        keyword = str(json[lbl]).split()
        # string mutation:
        try:
            df = df.replace({'€': ''}, regex=True)
        except:
            pass
        df[lbl] = df['text'].isin(keyword)
        if(not df[lbl].sum() >= len(keyword) ):
            not_found += 1
            
    return df, not_found

# add labels to df with fuzzy matching
def _label_dataframe_SCC(df, json):
    not_found = 0
    labels = json.keys()
    for lbl in labels:
        df[lbl] = False
        if lbl.endswith('-hoeveel'):
            continue
        if lbl == 'Template':
            continue
        if json[lbl] != json[lbl]:
            #print('label unavailable')
            continue

        keyword = str(json[lbl]).split()
        try:
            df = df.replace({'€': ''}, regex=True)
        except:
            pass
        for key in keyword:
            x = np.array([[fuzz.ratio(str(itm), key) > 80] for itm in df.text.array])
            frame = pd.DataFrame(x)
            df[lbl] = np.logical_or(df[lbl], frame[0])
            
        if(not df[lbl].sum() >= len(keyword) ):
            not_found += 1
    return df, not_found


# helper function
def set_start_end(df):  
    df['start'] = -1
    df['end'] = -1
    text = df['text'].to_numpy()
    text = [x for x in text if str(x) != 'nan']
    text = ' '.join(text).split()
    
    text_bounds = [[-1,-1]]
    
    # find character bounds
    index = 0
    for word in text:
        start = index
        index += len(str(word))
        end = index
        index += 1
        text_bounds = np.append(text_bounds, [[start, end]], axis=0)
    text_bounds = text_bounds[1:]
    index = 0
    
    for i, row in df.iterrows():
        if index >= len(text):
            return df
        if text[index] == row['text']:
            df['start'][i] = text_bounds[index][0]
            df['end'][i] = text_bounds[index][1]
            index += 1
    return df
            

# helper function
def set_offset(df):
    length = 0
    df['offset'] = 0
    for i, row in df.iterrows():
        if(str(row['text']) != 'nan'):
            df['offset'][i] = length
            length += len(row['text']) + 1
        else:
            df['offset'][i] = -1
    return df


# helper function:
def load_json(path):
    try:
        string = open(path, 'r').read()
    except:
        string = open(path[:-7] + '.txt', 'r').read()
    return json.loads(string)

# helper function
def _save_output(output, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        np.save(f, output)




if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='which data to preprocess (SROIE or CUSTOM)')
    args = parser.parse_args()
    
    if(args.dataset):
        if(args.dataset == "CUSTOM" or args.dataset == "SROIE" ):
            preprocess_data(args.dataset)
        else:
            print("unknown dataset name")
    else:
        preprocess_data("CUSTOM")
        preprocess_data("SROIE")
    
    
    
    
    
    
    
    
    
    
    
    