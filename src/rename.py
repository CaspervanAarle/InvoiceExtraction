# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:01:46 2020

Helper file to remove specific labels from the json dictionaries

@author: Casper
"""
import os
import json
    
def load_json(path):
    try:
        string = open(path, 'r').read()
    except:
        print("failed to open json file")
    return json.loads(string)

def _save_output(dictio, path):
    with open(path, 'w') as fp:
        json.dump(dictio, fp)

if __name__ == '__main__':
    path_in = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\invoicenet\\InvoiceNet\\ds_custom_newname2\\"
    path_out = "C:\\Users\\Casper\\Projects\\Topicus\\implementations\\invoicenet\\InvoiceNet\\ds_custom_newname4\\"
    filenames_json = [x for x in os.listdir(path_in) if x.endswith('.json')]
    for fn_json in filenames_json: 
        # load
        dictio = load_json(path_in + fn_json)
        new_dict = {}
        # change
        keys_values = dictio.items()
        for key, value in keys_values:
            if key == "Item1-hoeveel":
                continue
            if key == "Item2-hoeveel":
                continue
            if key == "Item3-hoeveel":
                continue
            if dictio[key] == "nan":
                continue
            else:
                new_dict[key] = value
        
        # save
        _save_output(new_dict, path_out + fn_json)