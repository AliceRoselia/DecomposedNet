# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:48:45 2026

@author: User
"""
import itertools
from collections import Counter
import numpy as np
import pickle


piece_data_formatted = np.load("../../Piece_data_formatted.npy")
best_qs = np.load("../../best_qs.npy")

print("Data loading complete.")

#First is 0,53
#The next is 6,13
#The next is (3, 56)
CUTOFF = 800000

def build_counter(piece_data_formatted, filter_vals = None):
    if filter_vals is None:
        filter_vals = []
        start = 0
    else:
        start = filter_vals[-1][1]+1
    print("Building a counter for:",filter_vals)
    

    piece_data_formatted_test = [i for i in piece_data_formatted if all(i[square] == piece for piece,square in filter_vals)]
    commons = Counter(itertools.chain.from_iterable(((*filter_vals,(int(piece),square),)
    for square, piece in enumerate(x[start:],start=start) if piece != -1) for x in piece_data_formatted_test) )
    
    return commons

first_common = build_counter(piece_data_formatted)

print("commons_dict_building_complete.")

#common_filter = [psq for psq,count in first_common.items() if count > CUTOFF]

#piece_data_formatted = [i for i in piece_data_formatted if any(sum(i[square] == piece for piece,square in filter_x) >= 2 for filter_x in common_filter)]

#print("Piece data filtered.")

second_commons = [build_counter(piece_data_formatted,psq) for psq,count in first_common.items() if count > CUTOFF]
second_commons = sum(second_commons,start = Counter())

#common_filter = [psq for psq,count in second_commons.items() if count > CUTOFF]
#piece_data_formatted = [i for i in piece_data_formatted if any(sum(i[square] == piece for piece,square in filter_x) >= 3 for filter_x in common_filter)]

#print("Piece data filtered.")

third_commons = [build_counter(piece_data_formatted,psq) for psq,count in second_commons.items() if count > CUTOFF]
third_commons = sum(third_commons,start = Counter())

with open("commons.pkl","wb") as f: 
    pickle.dump([first_common,second_commons,third_commons],f)