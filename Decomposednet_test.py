# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:05:54 2025

@author: User
"""

import numpy as np


piece_data_formatted = np.load("Piece_data_formatted.npy")
best_qs = np.load("best_qs.npy")


import torch
import torch.nn as nn
import torch.nn.functional as func
from adabelief_pytorch import AdaBelief

torch.manual_seed(42)


batch_size = test_batch_size = 64000

class subNet(nn.Module):
    def __init__(self, input_dim):
        
        super(subNet, self).__init__()
        self.feature_Transformer = nn.Linear(input_dim,2048)
        #self.input_bias = nn.Parameter(torch.randn(2048))
        self.first_layer = nn.Linear(1024, 16)
        self.out_layer = nn.Linear(32, 256)
        self.batch_size = batch_size
    def forward(self,x):
        
        
        
        transformed_feature = self.feature_Transformer(x.view(self.batch_size,-1))
        #torch.where( x.view(32,-1).expand(-1,-1,2048)
        #transformed_feature = torch.clamp(torch.stack(self.feature_Transformer[i].sum(dim=0) for i in x.view(32,-1)) + self.input_bias,0,1)
        #transformed_feature = torch.randn(32,2048).to("cuda")
        
        intermediate_feature = transformed_feature[:,:1024] * transformed_feature[:,1024:] * 127/128
        
        late_intermediate_feature = self.first_layer(intermediate_feature)
        
        return self.out_layer(torch.clamp(torch.concatenate((late_intermediate_feature,late_intermediate_feature*late_intermediate_feature*127/128),dim=1),0,1))
        
def set_mask(mask,*idxs):
    for i in idxs:
        j = i*64
        mask[j:j+64] = 1
        mask[j+384:j+448] = 1


class singlePerspectiveNet(nn.Module):
    def __init__(self):
        super(singlePerspectiveNet, self).__init__()
        self.pawn_subnet = subNet(128) # 128 features for pawns.
        self.minor_piece_subnet = subNet(256) #256 features for knights and bishops.
        self.major_piece_subnet = subNet(384) #384 features for rooks, queen, and king.
        self.diagonal_moving_piece_subnet = subNet(384) #384 features for bishops, queen, and king.
        self.combined_net = nn.Linear(768,256,bias=False)
        
        pawn_mask = torch.zeros(768,dtype = torch.bool)
        minor_piece_mask = torch.zeros(768,dtype = torch.bool)
        major_piece_mask = torch.zeros(768,dtype = torch.bool)
        diagonal_moving_piece_mask = torch.zeros(768,dtype = torch.bool)
        # pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
        set_mask(pawn_mask,0)
        set_mask(minor_piece_mask,1,2)
        set_mask(major_piece_mask, 3,4,5)
        set_mask(diagonal_moving_piece_mask, 2,4,5)
        
        self.pawn_mask = pawn_mask.expand(batch_size,-1)
        self.minor_piece_mask = minor_piece_mask.expand(batch_size,-1)
        self.major_piece_mask = major_piece_mask.expand(batch_size,-1)
        self.diagonal_moving_piece_mask = diagonal_moving_piece_mask.expand(batch_size,-1)
    
    def forward(self,x):
        pawn_feature = self.pawn_subnet(x[self.pawn_mask])
        minor_piece_feature = self.minor_piece_subnet(x[self.minor_piece_mask])
        major_piece_feature = self.major_piece_subnet(x[self.major_piece_mask])
        diagonal_piece_feature = self.diagonal_moving_piece_subnet(x[self.diagonal_moving_piece_mask])
        combined_feature = self.combined_net(x)
        
        return pawn_feature +minor_piece_feature + major_piece_feature + diagonal_piece_feature + combined_feature
        #return torch.concatenate((pawn_feature,minor_piece_feature,major_piece_feature,diagonal_piece_feature))
        

class DecomposedNet(nn.Module):
    def __init__(self):
        super(DecomposedNet, self).__init__()
        self.singlePerspectiveNet = singlePerspectiveNet()
        self.input_bias = nn.Parameter(torch.randn(512))
        self.linear1 = nn.Linear(512,32)
        self.linear2 = nn.Linear(64,32)
        
        self.linear_out = nn.Linear(32,1)
        self.skip_out = nn.Linear(512,1,bias=False)
        
    
    def forward(self,x):
        #transformed_features = []
        #for i in range(32):
        transformed_features = torch.concatenate((self.singlePerspectiveNet(x[:,:768]),self.singlePerspectiveNet(x[:,768:])),dim=1)
        #transformed_features.append(transformed_feature)
        
        
        
        stacked_features = torch.clamp(transformed_features+self.input_bias,0,1)
        linear1_out = self.linear1(stacked_features)
        
        
        linear1_activated = torch.clamp(torch.concatenate((linear1_out,linear1_out*linear1_out*127/128),dim=1),0,1)
        
        linear2_out = torch.clamp(self.linear2(linear1_activated),0,1)
        
        
        
        
        final_output = self.linear_out(linear2_out) + self.skip_out(stacked_features)
        
        return final_output.view(-1)*600
        
        
    


p = np.random.permutation(np.arange(4000000,len(piece_data_formatted)))[:batch_size]

piece_data_formatted = piece_data_formatted[p]
best_qs = best_qs[p]


q = torch.Tensor((best_qs+1)/2).to("cuda")

model = DecomposedNet()

model.load_state_dict(torch.load("DecomposedNet_v2_large_epoch6.pt",weights_only=True))
#model.skip_out.requires_grad_(False)
#model.skip_out.weight = nn.Parameter(torch.clamp(model.skip_out.weight,-1.0,1.0)) #So we can quantize it. 

model = model.to("cuda")




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