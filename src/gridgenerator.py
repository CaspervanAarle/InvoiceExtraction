# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:53:17 2020

@author: Casper
"""
import numpy as np

# generate the input and output of a dataframe for the neural network
def generate_io_grid(df, gridsize):
    """ generates an input and output grid for an invoice of the CUSTOM dataset """
    labels = ['Factuurdatum', 'Factuurnummer', 'Uitschrijver', 'Bedrag', 'Item1-naam', 'Item1-prijs', 'Item2-naam', 'Item2-prijs', 'Item3-naam', 'Item3-prijs']
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
        for idx, lbl in enumerate(labels):
            try:
                o_grid[xa][ya][idx] = int(row[lbl])
            except:
                pass
    o_grid = _add_dont_care(o_grid)
    return (word_grid, o_grid)


# generate the input and output of a dataframe for the neural network
def generate_io_grid_SROIE(df, gridsize):
    """ generates an input and output grid for an invoice of the SROIE dataset """
    labels = ['company', 'date', 'address', 'total']
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
        for idx, lbl in enumerate(labels):
            try:
                o_grid[xa][ya][idx] = int(row[lbl])
            except:
                pass
    o_grid = _add_dont_care(o_grid)
    return (word_grid, o_grid)


# generate only the input for the neural network
def generate_i_grid(df, gridsize):
    """ generates only the input grid; applicable to CUSTOM and SROIE dataset"""
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
    """ output field distribution must sum to 1. Add a final 'don't care' class"""
    care = 1*(array.sum(axis=2) == 1)
    dontcare = 1 - care
    dontcare = np.expand_dims(dontcare, axis=-1)
    return np.concatenate((array, dontcare), axis=-1)
