# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:15:45 2026

@author: User
"""

import pickle


with open("commons.pkl","rb") as f:
    loaded = pickle.load(f)
CUTOFF = 1000000    

for i in range(len(loaded)):
    loaded[i] = {i:j for i,j in loaded[i].items() if j > CUTOFF}
            
print(loaded)