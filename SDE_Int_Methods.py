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
import matplotlib.pyplot as plt


## STOCHASTIC RK2 AND RK4

class SDE_IntMethods:
    
    def __init__(self,f,g,sigmas,dt):
        
        self.dt=torch.tensor(dt,device=device)
        self.f=f
        self.g=g
        self.device='cuda'
        self.sigmas=sigmas
        self.N_noise=sigmas.size()[1]

        self.RK=RK
        

    def RK4(self,X,I,t):
        
        Tilde=(self.sigmas.unsqueeze(0))*torch.randn([X.size()[0],X.size()[1],self.N_noise],device=device)
        
        k1=self.f(X,t,I[:,:])
        l1=self.g(X,t,I[:,:])
                
        k1_=k1+torch.sqrt(2/self.dt)*torch.einsum('ijk,ijk->ij',l1,Tilde)
        
        k2=self.f(X+k1_*self.dt/2,t+self.dt/2,I[:,:])
        l2=self.g(X+k1_*self.dt/2,t+self.dt/2,I[:,:])
        k2_=k2+torch.sqrt(2/self.dt)*torch.einsum('ijk,ijk->ij',l2,Tilde)
        
        k3=self.f(X+k2_*self.dt/2,t+self.dt/2,I[:,:])
        l3=self.g(X+k2_*self.dt/2,t+self.dt/2,I[:,:])
        k3_=k3+torch.sqrt(1/self.dt)*torch.einsum('ijk,ijk->ij',l3,Tilde)
        
        k4=self.f(X+k3_*self.dt,t+self.dt,I[:,:])
        l4=self.g(X+k3_*self.dt,t+self.dt,I[:,:])
        
        x_new=X+1/6*(k1+2*k2+2*k3+k4)*self.dt+torch.sqrt(self.dt)*torch.einsum('ijk,ijk->ij', 1/6*(l1+2*l2+2*l3+l4),Tilde)
        
        return x_new

    def RK2(self,X,I,t):
        
        
        Tilde=(self.sigmas.unsqueeze(0))*torch.randn([X.size()[0],X.size()[1],self.N_noise],device=device)
        
        k1=self.f(X,t,I[:,:])
        l1=self.g(X,t,I[:,:])
        
        
        k1_=k1+torch.sqrt(1/self.dt)*torch.einsum('ijk,ijk->ij',l1,Tilde)
        
        k2=self.f(X+k1_*self.dt,t+self.dt,I[:,:])
        l2=self.g(X+k1_*self.dt,t+self.dt,I[:,:])
        
        x_new=X+1/2*(k1+k2)*self.dt+torch.sqrt(self.dt)*torch.einsum('ijk,ijk->ij', 1/2*(l1+l2),Tilde)
        
        return x_new

    
    def Compute_Dynamics(self,Input,x0,t0):
        
        T=Input.size()[2]
        batch_size=x0.size()[0]
        N=x0.size()[1]
        
        X=torch.zeros([batch_size,N,T]).to(self.device)
        X[:,:,0]=x0
        
        t=t0
        
        if self.RK==4:

            for n in range(1,T):
                
                I=Input[:,:,n]

                X[:,:,n]=self.RK4(X[:,:,n-1],I,t)

        if self.RK==2:

            for n in range(1,T):
                
                I=Input[:,:,n]

                X[:,:,n]=self.RK2(X[:,:,n-1],I,t)

        return X
    
## Standard RK2 and RK4

class ODE_IntMethods:
    
    def __init__(self,f,dt):
        
        self.dt=torch.tensor(dt,device=device)
        self.f=f
        self.device='cuda'
       
        self.RK=RK
        

    def RK4(self,X,I,t):
        
        k1=self.f(X,t,I[:,:])
        
        k2=self.f(X+k1*self.dt/2,t+self.dt/2,I[:,:])
        
        k3=self.f(X+k2*self.dt/2,t+self.dt/2,I[:,:])
        
        k4=self.f(X+k3*self.dt,t+self.dt,I[:,:])
        
        x_new=X+1/6*(k1+2*k2+2*k3+k4)*self.dt
        
        return x_new

    def RK2(self,X,I,t):
        
        
        k1=self.f(X,t,I[:,:])
         
        k2=self.f(X+k1*self.dt,t+self.dt,I[:,:])
       
        x_new=X+1/2*(k1+k2)*self.dt
        
        return x_new

    
    def Compute_Dynamics(self,Input,x0,t0):
        
        T=Input.size()[2]
        batch_size=x0.size()[0]
        N=x0.size()[1]
        
        X=torch.zeros([batch_size,N,T]).to(self.device)
        X[:,:,0]=x0
        
        t=t0
        
        if self.RK==4:

            for n in range(1,T):
                
                I=Input[:,:,n]

                X[:,:,n]=self.RK4(X[:,:,n-1],I,t)

        if self.RK==2:

            for n in range(1,T):
                
                I=Input[:,:,n]

                X[:,:,n]=self.RK2(X[:,:,n-1],I,t)

        return X
        

## CLASS TO GENERATE THE DATA FOR THE ANALYTICAL SYSTEMS
class Generate_Data:
    def __init__(self,f,g,Preprocess_S,sigmas):
        
        self.dt=dt
        self.f=f
        self.g=g
        self.device='cuda'
        self.sigmas=sigmas
        self.N_noise=sigmas.size()[1]
        
        self.NX_dim=NX_dim
        self.N_delays=N_delays
        
        self.SD=SDE_IntMethods(f,g,sigmas,dt)
        self.Preprocess_S=Preprocess_S
        
    def Generate_RandSignal(self,Amplitude,T_steps,Repeat,N_seq):
        
        N_types=np.shape(T_steps)[0]
        S_data=torch.zeros([np.sum(N_seq),N_S,T_steps[0]*Repeat[0]+self.N_delays+1],device=device)
        
        for k in range(N_types):
            
            T=T_steps[k]*Repeat[k]+1
            s=Amplitude*(2*torch.rand([N_seq[k],T_steps[k]],device=device)-1)
            S=torch.zeros([N_seq[k],T_steps[k]*Repeat[k]],device=device)

            for n in range(N_seq[k]):

                S[n,:]=torch.reshape(torch.transpose(torch.tile(s[n,:],[Repeat[k],1]),0,1),[1,-1])

            S=torch.concat([torch.tile(S[:,0].unsqueeze(1),[1,self.N_delays+1]),S],1)
            S=torch.tile(S.unsqueeze(1),[1,N_S,1])
            
            ind=np.int32(np.concatenate([np.zeros([1]),np.cumsum(N_seq)],0))
            S_data[ind[k]:ind[k+1],:,:]=torch.clone(S)
            
        
        return S_data
    
    
    def Generate_ConstantOrderedSignal(self,Amplitude,Repeat_seq,N_diff_seq):
        
        T=100
        S=torch.tile( torch.linspace(-Amplitude,Amplitude,N_diff_seq).unsqueeze(1).unsqueeze(2),[1,N_S,T+self.N_delays+1] ).to(device)
        
        S_test=torch.zeros([N_diff_seq*Repeat_seq,N_S,T+self.N_delays+1],device=device)
        
        for n in range(N_diff_seq):
            
            S_test[n*Repeat_seq:(n+1)*Repeat_seq,:,:]=torch.tile(S[n,:,:].unsqueeze(0),[Repeat_seq,1,1])
        
        return S_test
    
    
    def Generate_TestSignal(self,Amplitude,T_steps,Repeat,Repeat_seq,N_diff_seq):
        
        S_=self.Generate_RandSignal(Amplitude,T_steps,Repeat,N_diff_seq)
        
        N_seq=np.sum(N_diff_seq)
        S_test=torch.zeros([N_seq*Repeat_seq,N_S,T_steps[0]*Repeat[0]+self.N_delays+1],device=device)
        
        for n in range(N_seq):
            
            S_test[n*Repeat_seq:(n+1)*Repeat_seq,:,:]=torch.tile(S_[n,:,:].unsqueeze(0),[Repeat_seq,1,1])
        
        return S_test
        
        
    def Generate_D(self,S,X0,t0):
        
        X=self.SD.Compute_Dynamics(S,X0,t0)
                
        T=X.size()[2]
        X_data=torch.zeros([X.size()[0],self.NX_dim*(self.N_delays+1),T-self.N_delays-1],device=device)
        Target=torch.zeros([X.size()[0],self.NX_dim*(self.N_delays+1),T-self.N_delays-1],device=device)
        
            
        for t in range(self.N_delays,T-1):
            
            X_data[:,:,t-self.N_delays]=torch.reshape(X[:,0:self.NX_dim,t-self.N_delays:t+1].flip(2)\
                                                        .transpose(1,2),\
                                                        [X.size()[0],self.NX_dim*(self.N_delays+1)])
            Target[:,:,t-self.N_delays]=torch.reshape(X[:,0:self.NX_dim,t-self.N_delays+1:t+2].flip(2)\
                                                        .transpose(1,2),\
                                                        [X.size()[0],self.NX_dim*(self.N_delays+1)])
        
    
        Input=torch.concat([S[:,:,self.N_delays+1:],X_data],1)
        
        return X_data, Input, Target
    
    
    def forward(self,S,X0,t0):
        
        X=self.SD.Compute_Dynamics(S,X0,t0,Include_X0=False)
        T=X.size()[2]
        X_data=torch.zeros([X.size()[0],self.NX_dim*(self.N_delays+1),T-self.N_delays],device=device)
        
        for t in range(self.N_delays+1,T+1):
            
            X_data[:,:,t-self.N_delays-1]=torch.reshape(X[:,0:self.NX_dim,t-self.N_delays-1:t].flip(2)\
                                                        .transpose(1,2),\
                                                        [X.size()[0],self.NX_dim*(self.N_delays+1)])
            
         
        return X_data
    
        
    def Generate_DataSet(self,X0_start,Amplitude,T_steps,Repeat,Plot=True):
        
        
        N_seq=[5000,5000]

        ## Generate Train Data
        S=self.Generate_RandSignal(Amplitude,T_steps,Repeat,N_seq)
        
        S=self.Preprocess_S(S)
                
        X0=torch.concat([X0_start.unsqueeze(0).tile([S.size()[0],1]),torch.zeros([S.size()[0],N_aux])],1)
        t0=0.
        
        X, Input, Target=self.Generate_D(S,X0,t0)


        ## Generate Test Data
        Repeat_seq=1000
        N_diff_seq=[30,30]
        S_te=self.Generate_TestSignal(Amplitude,T_steps,Repeat,Repeat_seq,N_diff_seq)
        
        S_te=self.Preprocess_S(S_te)
        
        X0=torch.concat([X0_start.unsqueeze(0).tile([S_te.size()[0],1]),torch.zeros([S_te.size()[0],N_aux])],1)
        t0=0.

        X_te, Input_te, Target_te=self.Generate_D(S_te,X0,t0)

        S_te=torch.reshape(S_te,[np.sum(N_diff_seq,0),Repeat_seq,S_te.size()[1],-1])
        X_te=torch.reshape(X_te,[np.sum(N_diff_seq,0),Repeat_seq,X_te.size()[1],-1])
        Input_te=torch.reshape(Input_te,[np.sum(N_diff_seq,0),Repeat_seq,Input_te.size()[1],-1])
        Target_te=torch.reshape(Target_te,[np.sum(N_diff_seq,0),Repeat_seq,Target_te.size()[1],-1])
        
        
        ## Generate Validation Data
        Repeat_seq=1000
        N_diff_seq=[30,30]
        S_val=self.Generate_TestSignal(Amplitude,T_steps,Repeat,Repeat_seq,N_diff_seq)
        
        S_val=self.Preprocess_S(S_val)
        
        X0=torch.concat([X0_start.unsqueeze(0).tile([S_val.size()[0],1]),torch.zeros([S_val.size()[0],N_aux])],1)
        t0=0.

        X_val, Input_val, Target_val=self.Generate_D(S_val,X0,t0)

        S_val=torch.reshape(S_val,[np.sum(N_diff_seq,0),Repeat_seq,S_val.size()[1],-1])
        X_val=torch.reshape(X_val,[np.sum(N_diff_seq,0),Repeat_seq,X_val.size()[1],-1])
        Input_val=torch.reshape(Input_val,[np.sum(N_diff_seq,0),Repeat_seq,Input_val.size()[1],-1])
        Target_val=torch.reshape(Target_val,[np.sum(N_diff_seq,0),Repeat_seq,Target_val.size()[1],-1])

        
        ## Generate Ordered Data
        Repeat_seq=1000
        N_diff_seq=30
        S_te_ordered=self.Generate_ConstantOrderedSignal(Amplitude,Repeat_seq,N_diff_seq)
        
        S_te_ordered=self.Preprocess_S(S_te_ordered)
        
        X0=torch.concat([X0_start.unsqueeze(0).tile([S_te_ordered.size()[0],1]),torch.zeros([S_te_ordered.size()[0],N_aux])],1)
        t0=0.

        X_te_ordered, Input_te_ordered, Target_te_ordered=self.Generate_D(S_te_ordered,X0,t0)

        S_te_ordered=torch.reshape(S_te_ordered,[N_diff_seq,Repeat_seq,S_te_ordered.size()[1],-1])
        X_te_ordered=torch.reshape(X_te_ordered,[N_diff_seq,Repeat_seq,X_te_ordered.size()[1],-1])
        Input_te_ordered=torch.reshape(Input_te_ordered,[N_diff_seq,Repeat_seq,Input_te_ordered.size()[1],-1])
        Target_te_ordered=torch.reshape(Target_te_ordered,[N_diff_seq,Repeat_seq,Target_te_ordered.size()[1],-1])
        
        
        A1,A2,M1,M2,V1,V2,E=AutoCov(X_te_ordered,X_te_ordered)
        
        ## A few plots to check the class
        if Plot:
            
            T_plot=70
            fig, axs = plt.subplots(3, 3, figsize=(30,20))
            
            ############################
            ## PLOT INPUT, X, AND TARGETS
            
            ## Select indeces to plot
            ind=np.random.randint(0,Input.size()[0])
            ind_x=0
            
            
            ## Plot Input, Activity and Target
            l1,=axs[0,0].plot(Input[ind,ind_x,0:T_plot].detach().to('cpu'),'black',linewidth=2)
            l2,=axs[0,0].plot(X[ind,ind_x,0:T_plot].detach().to('cpu'),'r',linewidth=2)
            l3,=axs[0,0].plot(Target[ind,ind_x,0:T_plot].detach().to('cpu'),'b',linewidth=2)
            
            axs[0,0].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[0,0].legend([l1,l2,l3],["Input","X","Target"])
            axs[0,0].set_title("Input, X, Targets")
            
            ##############
            ## PLOT DELAYS
            
            ## Plot Input
            l1,=axs[0,2].plot(Input[ind,ind_x,0:T_plot].detach().to('cpu'),'black')
            
            ## Plot X
            for n in range(self.N_delays+1):

                l2,=axs[0,2].plot(X[ind,n*self.NX_dim,0:T_plot].detach().to('cpu'))
            
            
            axs[0,2].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[0,2].legend([l1,l2],["Input","Delay example"])
            axs[0,2].set_title("Input, X, Delays")
            
            ##########################
            ## PLOT INPUT, X AND TARGETS FOR TEST SET
            ind=np.random.randint(0,Input_te.size()[0])
            l1,=axs[0,1].plot(Input_te[ind,0,ind_x,0:T_plot].detach().to('cpu'),'black')
            l2,=axs[0,1].plot(X_te[ind,0,ind_x,0:T_plot].detach().to('cpu'),'--r')
            l3,=axs[0,1].plot(Target_te[ind,0,ind_x,0:T_plot].detach().to('cpu'),'--b')
            
            axs[0,1].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[0,1].legend([l1,l2,l3],["Input","X","Target"])
            axs[0,1].set_title("Input, X, Targets (Test set)")
            
            
            ##############
            ## PLOT DELAYS FOR TEST SET
            l1,=axs[1,2].plot(Input_te[ind,0,ind_x,0:T_plot].detach().to('cpu'),'black')
            
            for n in range(self.N_delays+1):

                l2,=axs[1,2].plot(X_te[ind,0,n*self.NX_dim,0:T_plot].detach().to('cpu'))
            
            
            axs[1,2].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[1,2].legend([l1,l2],["Input","Delay example"])
            axs[1,2].set_title("Input, X, Delays (Test set)")
            
            
            ############
            ## PLOT ACTIVITIES
            l1,=axs[1,1].plot(Input_te[ind,0,0,0:T_plot].detach().to('cpu'),'black',linewidth=2)
            
            N_p=int(Input_te.size()[1]/10)
            colors=plt.cm.magma(np.linspace(0, 1, N_p))
            
            l2,=axs[1,1].plot(X_te[ind,n,0,0:T_plot].detach().to('cpu'), color=colors[0])

            for n in range(N_p):
                axs[1,1].plot(X_te[ind,n,0,0:T_plot].detach().to('cpu'), color=colors[n], alpha=0.5)

            
            axs[1,1].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[1,1].legend([l1,l2],["Input","X"])
            axs[1,1].set_title("Input and X realisations (Test set)")
            
            
            ############
            ## PLOT ACTIVITIES example 2
            l1,=axs[1,0].plot(Input_te_ordered[0,0,0,0:T_plot].detach().to('cpu'),'black',linewidth=2)
            
            N_p=int(Input_te_ordered.size()[1]/10)
            colors=plt.cm.magma(np.linspace(0, 1, N_p))
            
            l2,=axs[1,0].plot(X_te_ordered[0,n,0,0:T_plot].detach().to('cpu'), color=colors[0])
                        
            for n in range(N_p):

                axs[1,0].plot(X_te_ordered[0,n,0,0:T_plot].detach().to('cpu'), color=colors[n])
            
            
            axs[1,0].plot(Input_te_ordered[-1,0,0,0:T_plot].detach().to('cpu'),'black',linewidth=2)
            
            colors=plt.cm.ocean(np.linspace(0, 1, N_p))

            for n in range(N_p):

                axs[1,0].plot(X_te_ordered[-1,n,0,0:T_plot].detach().to('cpu'), color=colors[n])
            
            axs[1,0].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[1,0].legend([l1,l2],["Input","X"])
            axs[1,0].set_title("Input and X realisations (Test set)")
            
            
            ############
            ## PLOT ACTIVITIES and DELAYS
            l1,=axs[2,2].plot(Input_te_ordered[-1,0,0,0:T_plot].detach().to('cpu'),'black')
            
            for n in range(self.N_delays):

                l2,=axs[2,2].plot(X_te_ordered[-1,0,n*self.NX_dim,0:T_plot].detach().to('cpu'))
                
            l3,=axs[2,2].plot(X_te_ordered[-1,0,self.N_delays*self.NX_dim,0:T_plot].detach().to('cpu'),'--')
            
            axs[2,2].set(xlabel='Step number', ylabel='Input and Dynamics')
            axs[2,2].legend([l1,l2],["Input","Delay example"])
            axs[2,2].set_title("Input, X, Delays (Test set)")
            
            
            axs[2,0].axis("off")
            
            ## PLOT VARIANCES
            
            axs[2,1].scatter(Input_te_ordered[:,0,0,-10].to('cpu'),V1[:,0,-10].to('cpu'),10,'black')
            l2=axs[2,1].scatter(Input_te_ordered[:,0,0,-10].to('cpu'),V2[:,0,-10].to('cpu'),10,'black')
            axs[2,1].scatter(Input_te_ordered[:,0,0,-2].to('cpu'),V2[:,0,-2].to('cpu'),10,'black')
            axs[2,1].scatter(Input_te_ordered[:,0,0,-20].to('cpu'),V2[:,0,-20].to('cpu'),10,'black')
            axs[2,1].set(xlabel='Input', ylabel='Variance of realisation')           

            
            return S, Input, Target, S_te, Input_te, Target_te, S_val, Input_val, Target_val, S_te_ordered, Input_te_ordered, Target_te_ordered        

        
        

def AutoCov(S1,S2):
    
    N_cases=S1.size()[0]
    batch_size=S1.size()[1]
    N_percase=int(batch_size/N_cases)
    N_dim=S1.size()[2]
    T=S1.size()[3]
    
    ACov1=torch.zeros([N_cases,N_dim,T,T],device=S1.device)
    ACov2=torch.zeros([N_cases,N_dim,T,T],device=S1.device)
    Means1=torch.zeros([N_cases,N_dim,T],device=S1.device)
    Means2=torch.zeros([N_cases,N_dim,T],device=S1.device)
    Var1=torch.zeros([N_cases,N_dim,T],device=S1.device)
    Var2=torch.zeros([N_cases,N_dim,T],device=S1.device)
    
    for n in range(N_cases):
        
        for m in range(N_dim):
            
            s1=S1[n,:,m,:]
            s2=S2[n,:,m,:]

            Means1[n,m,:]=torch.mean(s1,0)
            Means2[n,m,:]=torch.mean(s2,0)

            s1_m=(s1-torch.mean(s1,0).unsqueeze(0)).unsqueeze(2)
            s1_m_=(s1-torch.mean(s1,0).unsqueeze(0)).unsqueeze(1)

            s2_m=(s2-torch.mean(s2,0).unsqueeze(0)).unsqueeze(2)
            s2_m_=(s2-torch.mean(s2,0).unsqueeze(0)).unsqueeze(1)

            if s1.size()[0]>1 and s2.size()[0]>1:


                Var1[n,m,:]=torch.sqrt(torch.var(s1,0))
                Var2[n,m,:]=torch.sqrt(torch.var(s2,0))

            ACov1[n,m,:,:]=torch.mean(s1_m*s1_m_,0)/(Var1[n,m,:].unsqueeze(1)*Var1[n,m,:].unsqueeze(0)+0.001)
            ACov2[n,m,:,:]=torch.mean(s2_m*s2_m_,0)/(Var2[n,m,:].unsqueeze(1)*Var2[n,m,:].unsqueeze(0)+0.001)
    
    Errors=torch.zeros([3])
    
    MV1=torch.mean(Var1,2)
    MV2=torch.mean(Var2,2)
    
    ws=(MV1+MV2)/(torch.sum(MV1,0).unsqueeze(0)+torch.sum(MV2,0).unsqueeze(0)+0.001)
    
    Errors[0]=torch.mean(torch.sqrt(torch.pow(Means1-Means2,2)))
    Errors[1]=torch.mean(torch.sum(torch.sqrt(torch.pow(Var1-Var2,2))*ws.unsqueeze(2),0))
    Errors[2]=torch.mean(torch.sum(torch.pow(ACov1-ACov2,2)*ws.unsqueeze(2).unsqueeze(3),0))
   
    
    return ACov1, ACov2, Means1, Means2, Var1, Var2, Errors
    
    
    
    
    
