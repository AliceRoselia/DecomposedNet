# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 21:37:47 2026

@author: User
"""


import numpy as np


piece_data_formatted = np.load("../../Piece_data_formatted.npy")
best_qs = np.load("../../best_qs.npy")

import torch
import torch.nn as nn
import torch.nn.functional as func
from adabelief_pytorch import AdaBelief
from natten import na2d

torch.manual_seed(42)


batch_size = 4096







class NattenNet(nn.Module):
    def __init__(self):
        super(NattenNet, self).__init__()
        
    
        
    
    def forward(self,x):
        #transformed_features = []
        #for i in range(32):
        transformed_features = torch.concatenate((self.singlePerspectiveNet(x[:,:768]),self.singlePerspectiveNet(x[:,768:])),dim=1)
        #transformed_features.append(transformed_feature)
        
        
        
        stacked_features = torch.clamp(transformed_features+self.input_bias,0,1)
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
    #model.load_state_dict(torch.load("DecomposedNet_v2_large_epoch6.pt",weights_only=True))
    model = model.to("cuda")
    
    model = torch.compile(model)
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
        x = torch.zeros(batch_size, (64,12),dtype=torch.float)
        for j in range(batch_size):
            for k in range(64):
                piece_on_k = piece_data_formatted[i+j,k]
                if piece_on_k != -1:
                    x[k,int(piece_on_k)] = 1.0
    
    
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
            
    torch.save(model.state_dict(),"ChaoticNet_epoch1.pt")
    
#Epoch 1 result: 0.6265
#Epoch 2 result: 0.6250

#Epoch 4 result: 0.6195
#Epoch 5 result: 0.6192
#Epoch 6 result: 0.6189

