import os

import torch
import math
import random
from torch import nn
from modules import ConvSC, ConvNeXt_block, Learnable_Filter, Attention, ConvNeXt_bottle

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dim = 64

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 # half_dim = 32
        emb = math.log(10000) / (half_dim - 1) # emb = 0.2971077539347156
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # emb = (32,)
        emb = x[:, None] * emb[None, :] # emb = (16,32)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # emb =  (16,64)
        return emb

class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim*4) # dim= 64
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim*4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x) # x = (16,) ==> x = (16,64)
        x = self.linear1(x) # x = (16,64) ==> x = (16,256)
        x = self.gelu(x)
        x = self.linear2(x) #x = (16,256)  ==>  x = (16,64)
        return x

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x) # x = 160 * 1 * 64 * 64
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1 # latent = 160 * 64 * 16 * 16, enc1 = 160 * 64 * 64 * 64

class LP(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(LP,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(640, 64, 1)

    def forward(self, hid, enc1=None): # enc1 = 160 * 64 * 64 * 64
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid) # hid = 160 * 64 * 64 * 64
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1)) # Y = 160 * 64 * 64 * 64
        ys =Y.shape
        Y = Y.reshape(int(ys[0]/10), int(ys[1]*10), 64, 64) #  Y = 160 * 64 * 64 * 64 ==> Y = 16 * 640 * 64 * 64
        Y = self.readout(Y) ## Y = 16 * 640 * 64 * 64 ==> Y = 16 * 64 * 64 * 64
        return Y

class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super(Predictor, self).__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for i in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape  # B = 16, T = 20, C = 64, H = 16, W = 16 z =  16 * 20 * 64 * 16 * 16
        x = x.reshape(B, T*C, H, W) # x = 16 * 1280 * 16 * 16,time_emb=(16,64)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb) # z = 16 * 640 * 16 * 16

        y = z.reshape(B, int(T/2), C, H, W) # z = 16 * 640 * 16 * 16 ==>  y = 16 * 10 * 64 * 16 * 16
        return y

class IAM4VP(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        T, C, H, W = shape_in
        self.time_mlp = Time_MLP(dim=64)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T*hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.attn = Attention(64)
        self.readout = nn.Conv2d(64, 1, 1)
        self.mask_token = nn.Parameter(torch.zeros(10, hid_S, 16, 16))
        self.lp = LP(C, hid_S, N_S)

    def forward(self, x_raw, y_raw=None, t=None, is_train=None):
        B, T, C, H, W = x_raw.shape # B = 16, T = 10 , C = 1, H = 64, W = 64
        x = x_raw.view(B*T, C, H, W) # x_raw = 16 * 10 * 1 * 64 * 64 ==> x = 160 * 1 * 64 * 64
        time_emb = self.time_mlp(t)# t = (16,) ==> time_emb = (16 * 64)
        embed, skip = self.enc(x) # embed = 160 * 64 * 16 * 16, skip = 160 * 64 * 64 * 64
        mask_token = self.mask_token.repeat(B,1,1,1,1) #  self.mask_token = 10 * 64 * 16 * 16 ==> mask_token = 16 * 10 * 64 * 16 * 16

        for idx, pred in enumerate(y_raw):
            embed2,_ = self.lp(pred)
            mask_token[:,idx,:,:,:] = embed2

        _, C_, H_, W_ = embed.shape #  C_ = 64, H_ = 16 , W_ = 16

        z = embed.view(B, T, C_, H_, W_) # embed = 160 * 64 * 16 * 16 ==> z =  16 * 10 * 64 * 16 * 16
        z2 = mask_token # mask_token = 16 * 10 * 64 * 16 * 16
        z = torch.cat([z, z2], dim=1) # z =  16 * 20 * 64 * 16 * 16
        hid = self.hid(z, time_emb) # z =  16 * 20 * 64 * 16 * 16 time_emb = (16 * 64) == > hid  = 16 * 10 * 64 * 16 * 16
        hid = hid.reshape(B*T, C_, H_, W_) #  hid  = 160 * 64 * 16 * 16

        Y = self.dec(hid, skip) # hid = 160 * 64 * 16 * 16,skip = 160 * 64 * 64 * 64  ==>  Y = 16 * 64 * 64 * 64
        Y = self.attn(Y) #  Y = 16 * 64 * 64 * 64
        Y = self.readout(Y) #  Y = 16 * 64 * 64 * 64 ==> #  Y = 16 * 1 * 64 * 64
        return Y

    def save(self, itr, dir):
        stats = {'net_param': self.state_dict()}
        checkpoint_path = os.path.join(dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)

    def load(self, pm_checkpoint_path, device):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(device))
        self.load_state_dict(stats['net_param'])
if __name__ == "__main__":
    import numpy as np
    model = IAM4VP([10,1,64,64])
    inputs = torch.randn(2,10, 1,64,64)
    inputs2 = torch.randn(2,10, 1,64,64)
    pred_list = []
    for timestep in range(10):
        t = torch.tensor(timestep*100).repeat(inputs.shape[0])
        out = model(inputs, y_raw=pred_list, t=t)
        pred_list.append(out)
