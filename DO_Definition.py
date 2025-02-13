
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


def DO(x,t,I):
    
    alpha=-1
    beta=1
    delta=0.3
    
    y=torch.zeros([x.size()[0],x.size()[1]]).to(device)
    y[:,0]=x[:,1]

    y[:,1]=I[:,0]-delta*x[:,1]-alpha*x[:,0]-beta*x[:,0]**3
    
    ## colored noise
    if x.size()[1]>1:
        
        tau_noise1=0.2
        tau_noise2=0.8
        y[:,2]=-1/tau_noise1*x[:,2]
        y[:,3]=-1/tau_noise2*x[:,3]
        y[:,0]+=torch.sum(y[:,2:],1)
        
        
    return y


def Noise_SigXI(x,t,I):
    
    y_noise=torch.zeros([x.size()[0],x.size()[1],2]).to(device)
    
    ## colored noise
    if x.size()[1]>1:
    
        y_noise[:,1,0]=sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.002
        y_noise[:,2,0]=sigma_real*torch.tanh(torch.abs(3*x[:,0]))
        
    return y_noise

def Preprocess_S(S):
    
    T=S.size()[2]
    
    omega=1.2
    dt=0.09*(2*torch.pi/omega)
    
    S=S*torch.concat([torch.sin(omega*torch.arange(T,device=device)*dt).unsqueeze(0)])

    return S
    
