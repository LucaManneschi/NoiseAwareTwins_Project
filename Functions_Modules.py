import numpy as np

import torch
from torch import nn

# from torch.nn.utils import weight_norm



class G2F_Module(nn.Module):
    
    def __init__(self, N_aux, N_ext_aux, NX_dim, device):
        super().__init__()
        
        self.N_aux=N_aux
        self.N_ext_aux=N_ext_aux
        self.NX_dim=NX_dim
        self.device = device
        
        self.W_noise=nn.Parameter(torch.ones([N_aux, self.NX_dim], device=self.device))
        self.W_ext=nn.Parameter(torch.ones([N_ext_aux, self.NX_dim], device=self.device))
                
        theta_noise=np.float32(0.7**(np.arange(0,self.N_aux))*5) 
        theta_noise_ext=np.float32(0.7**(np.arange(0,self.N_ext_aux))*5) 

        theta_noise=np.concatenate([theta_noise,theta_noise_ext],0)


        self.Theta_noise=nn.Parameter(torch.tensor(theta_noise, device=self.device))

            
    def forward(self, y, X_aux):
                    
        X_noise=-torch.abs(self.Theta_noise).unsqueeze(0)*X_aux
            
        W_noise=self.N_aux*torch.softmax(self.W_noise,0)
        
        y[:,0:self.NX_dim]=y[:,0:self.NX_dim]+torch.matmul(X_noise[:,0:self.N_aux],W_noise)
        
        y=torch.concat([y,X_noise],1)
        
        return y
        
        
        

class F_Module(nn.Module):
    
    def __init__(self, F_Ns, N_aux, N_ext_aux, load_weights, NX_dim, device):
        super().__init__()
                
        self.device = device

        module=[]
        if len(load_weights)==0:
            for n in range(1,F_Ns.size()[0]):
                            
                module.append(nn.Linear(F_Ns[n-1],F_Ns[n]))
            
                with torch.no_grad():
                    
                    module[-1].bias[:]=torch.tensor(0.)
                    module[-1].weight[:,:]=torch.randn(module[-1].weight.size(),device=self.device)*0.02
                
                if n<F_Ns.size()[0]-1:
                
                    module.append(nn.ReLU())
        else:
            for n in range(1,F_Ns.size()[0]):
                            
                module.append(nn.Linear(F_Ns[n-1],F_Ns[n]))
            
                with torch.no_grad():
                    
                    module[-1].bias[:]=load_weights[1][n-1]
                    module[-1].weight[:,:]=load_weights[0][n-1].T
                
                if n<F_Ns.size()[0]-1:
                
                    module.append(nn.ReLU())
        self.F=nn.Sequential(*module)
        
        self.F_Ns=F_Ns
        self.N_aux=N_aux
        self.N_ext_aux=N_ext_aux
        self.NX_dim=NX_dim
        
        if (self.N_aux+self.N_ext_aux)>0:
            
            self.G2F=G2F_Module(N_aux, N_ext_aux, NX_dim, device)
                    
    def forward(self, X, t, Input):
        
        if self.N_aux==0 and self.N_ext_aux==0:
            
            s=torch.concat([Input,X],1)
            
        else:
            
            X_f=X[:,0:self.F_Ns[-1]]
            X_aux=X[:,self.F_Ns[-1]:]
            
                        
            s=torch.concat([Input,X_f],1)
            
        y=self.F(s)
                
        if (self.N_aux+self.N_ext_aux)>0:
            
            y=self.G2F(y.clone(), X_aux.clone())
            
        return y
    
    

class G_Module(nn.Module):
    
    def __init__(self, G_Ns, F_Ns, N_aux, sigmas_model, device):
        super().__init__()
        
        module=[]
        
        for n in range(1,G_Ns.size()[0]):
                        
            module.append(nn.Linear(G_Ns[n-1],G_Ns[n]))
        
            with torch.no_grad():
                
                module[-1].bias[:]=torch.tensor(0.)
                module[-1].weight[:,:]=torch.randn(module[-1].weight.size(),device=device)*0.02
            
            if n<G_Ns.size()[0]-1:
            
                module.append(nn.ReLU())
                
        self.G=nn.Sequential(*module)
        
        self.G_out=sigmas_model.size()[0]
        self.G_Ns=G_Ns
        self.N_aux=N_aux
        self.F_Ns=F_Ns
        self.device = device
        
        
    def forward(self,X,t,Input):
        
        # xs=[]
        
        if self.N_aux==0:
        
            s=torch.concat([Input,X],1)
        
        else:
            
            X_f=X[:,0:self.F_Ns[-1]]
            
            s=torch.concat([Input,X_f],1)

        y=self.G(s)
        
        y=torch.reshape(y,[Input.size()[0],self.G_out,-1])
        
        return y
    

class D_Module(nn.Module):
    
    def __init__(self, D_Ns, load_weights, device = 'cpu'):
        super().__init__()
                
        self.device = device
        module=[]
        if len(load_weights)==0:
       
            for n in range(1,D_Ns.size()[0]-1):

                module.append(nn.Linear(D_Ns[n-1],D_Ns[n]))

                with torch.no_grad():

                    module[-1].bias[:]=torch.tensor(0.)
                    module[-1].weight[:,:]=torch.randn(module[-1].weight.size(),device=self.device)*0.02

                    module.append(nn.ReLU())   
        else:
            
            for n in range(1,D_Ns.size()[0]-1):
                            
                module.append(nn.Linear(D_Ns[n-1],D_Ns[n]))
            
                with torch.no_grad():
                    
                    module[-1].bias[:]=load_weights[1][n-1]
                    module[-1].weight[:,:]=load_weights[0][n-1].T
                                
                    module.append(nn.ReLU())  
                
        self.D=nn.Sequential(*module)
        self.out=nn.Linear(D_Ns[-2],D_Ns[-1])
        
        if len(load_weights)>0:
            with torch.no_grad():
            
                self.out.bias[:]=load_weights[1][-1]
                self.out.weight[:,:]=load_weights[0][-1].T
        
    def forward(self,S):
        
        Df=self.D(S)
        D=self.out(Df)
        
        return D, Df
        
        
