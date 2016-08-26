#!/usr/bin/env python 

import csv, shutil, os

source = '/home/james/Desktop/PAC Data/pac_2016_data_files'

dest = '/home/james/Desktop/PAC Data/pac_2016_data_holdout'

files = os.listdir(source)

with open('HC_holdout_list', 'r') as f:
    hclist =  list(csv.reader(f))
    
hclist2 = hclist[0]
hclist = hclist2

for file in hclist:
    file2 = source + '/'+ file
    shutil.move(file2, dest)
