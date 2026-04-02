# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 21:37:47 2026

@author: User
"""


import numpy as np


piece_data_formatted = np.load("../../Piece_data_formatted_2.npy")
best_qs = np.load("../../best_qs_2.npy")

import torch
import torch.nn as nn
import torch.nn.functional as func
from adabelief_pytorch import AdaBelief

torch.manual_seed(42)


batch_size = 4096


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
    
    
    

p = np.random.permutation(4000000)
piece_data_formatted = piece_data_formatted[p]
best_qs = best_qs[p]

q = torch.Tensor((best_qs+1)/2).to("cuda")



def flip_piece(piece):
    if piece>=6:
        return piece-6
    else:
        return piece+6

def flip_square(square):
    return 56-(square&56) + (square&7)


if __name__ == "__main__":
    # Create model
    model = ChaoticNet()
    model = model.to("cuda")
    model = torch.compile(model)
    model.load_state_dict(torch.load("ChaoticNet6_epoch1.pt",weights_only=True))
    #The first epoch was trained with lr=0.001 and weight_decay = 0
    #2nd epoch: weight_decay = 1e-4, lr = 0.0003
    #3rd epoch: weight decay = 3e-3, lr = 0.0003
    #4th epoch: Using new data. Same as 3rd epoch.
    #5th epoch: Using new data again.
    #6th epoch: New data again.
    #7th epoch: Not working.
    optimizer = AdaBelief(model.parameters(),lr=0.0003,weight_decouple=True,weight_decay=3e-3,rectify=False)
    # Create a sample sparse binary input (batch_size=2, input_dim=768)
    for i in range(0,3997696,batch_size):
        if i%1024 == 0:
            print(i)
        x = torch.zeros(batch_size, 768*2,dtype=torch.float)
        for j in range(batch_size):
            for k in range(64):
                piece_on_k = piece_data_formatted[i+j,k]
                if piece_on_k != -1:
                    index = int(piece_on_k*64 + k)
                    flipped_index = int(768+flip_piece(piece_on_k)*64+flip_square(k))
                    x[j, index] = 1
                    x[j, flipped_index] = 1
    
    
    # Forward pass
        output = model(x.to("cuda"))
        probability = func.sigmoid(output/410)
        loss = func.binary_cross_entropy(probability, q[i:i+batch_size])
        loss.backward()
        #ideal_loss = func.binary_cross_entropy(q[i:i+batch_size], q[i:i+batch_size])
        optimizer.step()
        #print(model.U.grad)
        #print(model.V.grad)
        #for x in model.parameters():
            #print(x.grad)
        optimizer.zero_grad()
        if i%1024 == 0:
            print("loss:", loss)
            #print("ideal loss", ideal_loss)
            
    torch.save(model.state_dict(),"ChaoticNet6_epoch2.pt")
    
#Epoch 1 result: 0.6265
#Epoch 2 result: 0.6250

#Epoch 4 result: 0.6195
#Epoch 5 result: 0.6192
#Epoch 6 result: 0.6189

#With simple 1d conv: 0.6284
#With a bit more 1d conv: 0.6264
#With a new setup: 0.6260
#Without all the messy convs: 0.6254

#With a few layers of self-convs. 0.6247
#Only 1 conv layer, but with double the input: 0.6259
#Epoch 2 with new data: 0.6227

