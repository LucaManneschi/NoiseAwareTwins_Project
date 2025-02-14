
import torch


class Leaky_Integrator:
    def __init__(self):

        self.sigma_real = 0.0

        self.tau = 1
        self.inv_tau = 1.0/self.tau
            
        self.tau_noise1 =0.2
        self.tau_noise2 =0.8

        self.inv_tau_noise1 = 1.0 / self.tau_noise1
        self.inv_tau_noise2 = 1.0 / self.tau_noise2

    def model(self, x, t, I):
        y=torch.zeros([x.size()[0],x.size()[1]]).to(x.device)
        
        y[:,0]=- self.inv_tau*x[:,0] + self.inv_tau * torch.tanh(I[:,0])
        
        ## colored noise
        if x.size()[1]>1:
            y[:,1]=-self.inv_tau_noise1*x[:,1]
            y[:,2]=-self.inv_tau_noise2*x[:,2]
            y[:,0]+=torch.sum(y[:,1:],1)
        return y



    def Noise_SigXI(self, x, t, I):
        
        y_noise=torch.zeros([x.size()[0],x.size()[1],2]).to(x.device)
        
        y_noise[:,0,0]=self.sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.02
        y_noise[:,0,1]=self.sigma_real*torch.tanh(torch.abs(3*x[:,0]))
        
        ## colored noise
        if x.size()[1]>1:
            y_noise[:,1,0]=self.sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.02
            y_noise[:,2,0]=self.sigma_real*torch.tanh(torch.abs(3*x[:,0]))
                
        return y_noise


    def Preprocess_S(self, S, device):
        return S

