
import torch

class Duffing_Oscillator:
    def __init__ (self):

        self.sigma_real = 0.05

        self.alpha=-1
        self.beta=1
        self.delta=0.3

        self.tau_noise1 =0.2
        self.tau_noise2 =0.8

        self.inv_tau_noise1 = 1.0 / self.tau_noise1
        self.inv_tau_noise2 = 1.0 / self.tau_noise2

        self.omega=1.2
        self.dt=0.09*(2*torch.pi/self.omega)

    def model(self, x, t, I):
        
        y=torch.zeros([x.size()[0],x.size()[1]]).to(x.device)
        y[:,0]=x[:,1]

        y[:,1]=I[:,0]  -self.delta*x[:,1] -self.alpha*x[:,0] -self.beta*x[:,0]**3
        
        ## colored noise
        if x.size()[1]>1:
            y[:,2] = -self.inv_tau_noise1*x[:,2]
            y[:,3] = -self.inv_tau_noise2*x[:,3]
            y[:,0] += torch.sum(y[:,2:],1)

        return y


    def Noise_SigXI(self, x, t, I):
        
        y_noise=torch.zeros([x.size()[0],x.size()[1],2]).to(x.device)
        
        ## colored noise
        if x.size()[1]>1:
            y_noise[:,1,0]=self.sigma_real*torch.tanh(torch.abs(2*I[:,0]))+0.002
            y_noise[:,2,0]=self.sigma_real*torch.tanh(torch.abs(3*x[:,0]))
            
        return y_noise

    def Preprocess_S(self, S, device):
        
        T=S.size()[2]
        
        S=S*torch.concat([torch.sin(self.omega*torch.arange(T, device=device)*self.dt).unsqueeze(0)])

        return S
        
