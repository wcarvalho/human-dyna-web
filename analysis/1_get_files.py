#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:35:59 2024

@author: hall-mcmaster
"""

# This script loads potential data files and checks:
    # 1. Their size
    # 2. Whether the dataset is 'complete'
    
# If the dataset is 'complete' (the participant completed all trials),
# the file is assigned a subject number of the format sub-*


# import packages
import os
from os.path import join as opj
import numpy as np
import json


# locate files in the data raw directory
def list_files(directory):
    fnames=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            cfile=opj(root,file)
            fnames.append(cfile)
    return fnames

cdir=os.getcwd()
project_dir=os.path.dirname(cdir)
maze_dir=os.path.dirname(project_dir)
#data_dir_raw=opj(maze_dir,'human-dyna-web-raw-data','pilot-1')
data_dir_raw=opj(maze_dir,'human-dyna-web-raw-data','pilot-2')
raw_files=list_files(data_dir_raw)

# create output folder for the subset of relevant files
#exp_string='pilot-1-subset'
exp_string='pilot-2-subset'
data_dir_output=opj(maze_dir,'human-dyna-web-raw-data', exp_string)
os.makedirs(data_dir_output, exist_ok=True)


# loop over files and check for 'complete' data sets
max_stage_possible=22 # hard coded for based on specific experiment
nfiles=len(raw_files)
file_paths_filtered=[]
sub_count=0

for file_path in raw_files:
    with open(file_path, 'r') as f:
        data_dicts = json.load(f)
        save_data=0
        
        # if examining data from pilot experiment 1
        if exp_string=='pilot-1-subset':
            # assess number of stages in data file
            stage_idx_values = [d['stage_idx'] for d in data_dicts]
            stage_idx_array = np.array(stage_idx_values)
            unique_stage_idx = np.unique(stage_idx_array)
            cmax_stage=np.max(unique_stage_idx)
            if cmax_stage>max_stage_possible:
                max_stage=cmax_stage
            if cmax_stage>=max_stage_possible:
                save_data=1
        
        # if examining data from later experiments       
        else:
            
            if np.size(data_dicts) == 0 or np.ndim(data_dicts) == 0:
                continue
            
            if data_dicts and data_dicts[-1].get('finished') == True:
                save_data=1
                        
                
        # if the number of changes are as expected, retain the file
        if save_data==1:
            file_paths_filtered.append(file_path)
            
            # update the filename and save (e.g. sub-01)
            sub_count+=1
            sub_string = f'sub-{sub_count:02}' + '.json'
            fname_new=opj(data_dir_output,sub_string)
            
            with open(fname_new, 'w') as file:
                json.dump(data_dicts, file)
                    

            
    


