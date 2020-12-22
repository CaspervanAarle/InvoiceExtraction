# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:53:17 2020

@author: Casper
"""
import numpy as np
import custom_tokenizer

# generate the input and output of a dataframe for the neural network
def generate_io_grid(df, gridsize):
    x,y = gridsize
    word_grid = [['' for _ in range(y)] for _ in range(x)] 
    o_grid = np.zeros((x,y,10))
    
    width = df['width'].max()
    height = df['height'].max()
    
    df = df.dropna() 
    
    for i, row in df.iterrows():
        xa =int(round(((int(row['left'])+int(row['width'])/2)/width)*(x-1)))
        ya =int(round(((int(row['top'])+int(row['height'])/2)/height)*(y-1)))
        
        # fill input words
        word_grid[xa][ya] = row['text']

        # fill output truths
        try:
            o_grid[xa][ya][0] = int(row['Factuurdatum'])
        except:
            pass
        try:
            o_grid[xa][ya][1] = int(row['Factuurnummer'])
        except:
            pass
        try:
            o_grid[xa][ya][2] = int(row['Uitschrijver'])
        except:
            pass
        try:
            o_grid[xa][ya][3] = int(row['Bedrag'])
        except:
            pass
        try:
            o_grid[xa][ya][4] = int(row['Item1-naam'])
        except:
            pass
        try:
            o_grid[xa][ya][5] = int(row['Item1-prijs'])
        except:
            pass
        try:
            o_grid[xa][ya][6] = int(row['Item2-naam'])
        except:
            pass
        try:
            o_grid[xa][ya][7] = int(row['Item2-prijs'])
        except:
            pass
        try:
            o_grid[xa][ya][8] = int(row['Item3-naam'])
        except:
            pass
        try:
            o_grid[xa][ya][9] = int(row['Item3-prijs'])  
        except:
            pass
        
    
    o_grid = _add_dont_care(o_grid)
    return (word_grid, o_grid)


# generate the input and output of a dataframe for the neural network
def generate_io_grid_SROIE(df, gridsize):
    #print(df)
    x,y = gridsize
    word_grid = [['' for _ in range(y)] for _ in range(x)] 
    o_grid = np.zeros((x,y,4))
    
    width = df['width'].max()
    height = df['height'].max()
    
    df = df.dropna()  
    
    for i, row in df.iterrows():
        xa =int(round(((int(row['left'])+int(row['width'])/2)/width)*(x-1)))
        ya =int(round(((int(row['top'])+int(row['height'])/2)/height)*(y-1)))
        
        # fill input words
        word_grid[xa][ya] = row['text']

        # fill output truths
        try:
            o_grid[xa][ya][0] = int(row['company'])
        except:
            pass
        try:
            o_grid[xa][ya][1] = int(row['date'])
        except:
            pass
        try:
            o_grid[xa][ya][2] = int(row['address'])
        except:
            pass
        try:
            o_grid[xa][ya][3] = int(row['total'])
        except:
            pass
    
    o_grid = _add_dont_care(o_grid)
    return (word_grid, o_grid)


# generate only the input for the neural network
def generate_i_grid(df, gridsize):
    x,y = gridsize
    word_grid = [['' for _ in range(y)] for _ in range(x)] 
    width = df['width'].max()
    height = df['height'].max()
    df = df.dropna()
    for i, row in df.iterrows():
        xa =int(round(((int(row['left'])+int(row['width'])/2)/width)*(x-1)))
        ya =int(round(((int(row['top'])+int(row['height'])/2)/height)*(y-1)))
        
        # fill input words
        word_grid[xa][ya] = row['text']
    return word_grid


# helper function
def _add_dont_care(array):
    care = 1*(array.sum(axis=2) == 1)
    dontcare = 1 - care
    dontcare = np.expand_dims(dontcare, axis=-1)
    return np.concatenate((array, dontcare), axis=-1)
