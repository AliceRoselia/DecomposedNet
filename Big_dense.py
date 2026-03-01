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

torch.manual_seed(42)


batch_size = 4096




class subNet(nn.Module):
    def __init__(self):
        super(subNet, self).__init__()
        self.linears = nn.ParameterList(
            [nn.Linear(1024,1024) for i in range(4)]
        )
    def forward(self,x):
        x_out = x
        for i in range(4):
            x_out = func.mish(self.linears[i](x_out))
        return x+x_out



class BigDenseNet(nn.Module):
    def __init__(self):
        super(BigDenseNet, self).__init__()
        self.into = nn.Linear(768,1024)
        self.subnets = nn.ParameterList(
            subNet() for i in range(18)
        )
        self.out = nn.Linear(1024,1)
    
        
    
    def forward(self,x):
        #transformed_features = []
        #for i in range(32):
        x = func.mish(self.into(torch.flatten(x,start_dim=1)))
        
        for i in range(18):
            x = self.subnets[i](x)
        
        final_outputs = self.out(x)
            
        
        return final_outputs.view(-1)/32
    
    
    

p = np.random.permutation(4000000)
piece_data_formatted = piece_data_formatted[p]
best_qs = best_qs[p]

q = torch.Tensor((best_qs+1)/2).to("cuda")




if __name__ == "__main__":
    # Create model
    model = BigDenseNet()
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
        x = torch.zeros(batch_size,64,12,dtype=torch.float)
        for j in range(batch_size):
            for k in range(64):
                piece_on_k = piece_data_formatted[i+j,k]
                if piece_on_k != -1:
                    x[j,k,int(piece_on_k)] = 1.0
    
    
    # Forward pass
        output = model(x.to("cuda"))
        probability = func.sigmoid(output*10)
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
            
    torch.save(model.state_dict(),"BigDenseNet_epoch1.pt")
    
#Epoch 1 result: 0.6265
#Epoch 2 result: 0.6250

#Epoch 4 result: 0.6195
#Epoch 5 result: 0.6192
#Epoch 6 result: 0.6189

