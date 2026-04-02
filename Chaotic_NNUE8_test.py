# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 22:04:55 2026

@author: User
"""

import numpy as np


piece_data_formatted = np.load("../../Piece_data_formatted_4.npy")
best_qs = np.load("../../best_qs_4.npy")


import torch
import torch.nn as nn
import torch.nn.functional as func
from adabelief_pytorch import AdaBelief

torch.manual_seed(42)


batch_size = test_batch_size = 256000

class singlePerspectiveNet(nn.Module):
    def __init__(self):
        super(singlePerspectiveNet, self).__init__()
        self.combined_net = nn.Linear(768,2048)
        self.conv3 = nn.Conv1d(8, 1, 1,padding = "same", padding_mode="circular")
        
        
        # pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
        
    
    def forward(self,x):
        combined_feature = self.combined_net(x).view(-1,8,256)
        combined_feature = torch.clamp(combined_feature,0,1)
        combined_feature = self.conv3(combined_feature).squeeze()
        
        return combined_feature
        #return torch.concatenate((pawn_feature,minor_piece_feature,major_piece_feature,diagonal_piece_feature))

class ChaoticNet(nn.Module):
    def __init__(self):
        super(ChaoticNet, self).__init__()
        self.singlePerspectiveNet = singlePerspectiveNet()
        self.linear1 = nn.Linear(512,32)
        self.linear2 = nn.Linear(64,32)
        
        self.linear_out = nn.Linear(32,1)
        self.skip_out = nn.Linear(512,1,bias=False)
        
    
    def forward(self,x):
        #transformed_features = []
        #for i in range(32):
        transformed_features = torch.concatenate((self.singlePerspectiveNet(x[:,:768]),self.singlePerspectiveNet(x[:,768:])),dim=1)
        #transformed_features.append(transformed_feature)
        
        
        
        stacked_features = torch.clamp(transformed_features,0,1)
        linear1_out = self.linear1(stacked_features)
        
        
        linear1_activated = torch.clamp(torch.concatenate((linear1_out,linear1_out*linear1_out*127/128),dim=1),0,1)
        
        linear2_out = torch.clamp(self.linear2(linear1_activated),0,1)
        
        
        
        
        final_output = self.skip_out(stacked_features) + self.linear_out(linear2_out)
        
        return final_output.view(-1)*600
        
    


p = np.random.permutation(np.arange(4000000,len(piece_data_formatted)))[:batch_size]

piece_data_formatted = piece_data_formatted[p]
best_qs = best_qs[p]


q = torch.Tensor((best_qs+1)/2).to("cuda")

model = ChaoticNet()

model = model.to("cuda")
model = torch.compile(model)

model.load_state_dict(torch.load("ChaoticNet8_epoch2.pt",weights_only=True))
#model.skip_out.requires_grad_(False)
#model.skip_out.weight = nn.Parameter(torch.clamp(model.skip_out.weight,-1.0,1.0)) #So we can quantize it. 





def flip_piece(piece):
    if piece>=6:
        return piece-6
    else:
        return piece+6

def flip_square(square):
    return 56-(square&56) + (square&7)

with torch.no_grad():
    loss = 0
    entropy = 0
    
    x = torch.zeros(test_batch_size, 768*2,dtype=torch.float)
    for j in range(test_batch_size):
        for k in range(64):
            piece_on_k = piece_data_formatted[j,k]
            if piece_on_k != -1:
                index = int(piece_on_k*64 + k)
                flipped_index = int(768+flip_piece(piece_on_k)*64+flip_square(k))
                x[j, index] = 1
                x[j, flipped_index] = 1
    output = model(x.to("cuda"))
    probability = func.sigmoid(output/410)
    loss = func.binary_cross_entropy(probability, q)
    entropy = func.binary_cross_entropy(q, q)
print(loss)
print(entropy)

import matplotlib.pyplot as plt
plt.scatter(output.cpu().detach().numpy()[:1000],q.cpu().numpy()[:1000])
plt.show()