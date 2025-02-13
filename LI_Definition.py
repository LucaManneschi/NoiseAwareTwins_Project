
import torch
import numpy as np
torch.cuda.is_available()

from torch import nn
from torch import optim
import os
import pickle
device='cuda'
torch.set_default_tensor_type(torch.FloatTensor)

from Setting_Variables import *

def LI(x,t,I):
    
    tau=1
    y=torch.zeros([x.size()[0],x.size()[1]]).to(device)
    
    y[:,0]=-1/tau*x[:,0]+1/tau*torch.tanh(I[:,0])
    
    ## colored noise
    if x.size()[1]>1:
        
        tau_noise1=0.2
        tau_noise2=0.8
        y[:,1]=-1/tau_noise1*x[:,1]
        y[:,2]=-1/tau_noise2*x[:,2]
        y[:,0]+=torch.sum(y[:,1:],1)
        
    
    return y



def Noise_SigXI(x,t,I):
    
    y_noise=torch.zeros([x.size()[0],x.size()[1],2]).to(device)
    
    y_noise[:,0,0]=sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.02
    y_noise[:,0,1]=sigma_real*torch.tanh(torch.abs(3*x[:,0]))
    
    ## colored noise
    if x.size()[1]>1:
        
        y_noise[:,1,0]=sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.02
        y_noise[:,2,0]=sigma_real*torch.tanh(torch.abs(3*x[:,0]))
            
    return y_noise


def Preprocess_S(S):
    
    return S

