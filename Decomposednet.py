# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:34:55 2025

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




batch_size = 4096

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
    model = DecomposedNet()
    model.load_state_dict(torch.load("DecomposedNet_v2_large_epoch4.pt",weights_only=True))
    model = model.to("cuda")
    
    model = torch.jit.script(model)
    #The first epoch was trained with lr=0.001 and weight_decay = 0
    #2nd epoch: weight_decay = 1e-4, lr = 0.0003
    #3rd epoch: weight decay = 3e-3, lr = 0.0003
    #4th epoch: Using new data. Same as 3rd epoch.
    #5th epoch: Still working in progress.
    optimizer = AdaBelief(model.parameters(),lr=0.0003,weight_decouple=True,weight_decay=3e-3, rectify=False)
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
            
    torch.save(model.state_dict(),"DecomposedNet_v2_large_epoch5.pt")
    
#Epoch 1 result: 0.6265
#Epoch 2 result: 0.6250

#Epoch 4 result: 0.6195