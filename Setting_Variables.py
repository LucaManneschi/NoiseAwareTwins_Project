import torch
device='cuda'


Dynamical_System='DO'

if Dynamical_System=='DO':
    
    
    sigma_real=torch.tensor(0.02,device=device)   ## Standard deviation of the noise present in the system (it will actually vary
                                                  ## following a function G, but this scales the total level of noise)
    
    N_aux=2                        ## Number of auxiliary variables of the analytical system (not the model)
    NX_dim=2                       ## Observable dimensionality of the Duffing oscillator (x and v) 
    N_noise=2                      ## Noise Dimensionality

    N_delays=2                     ## Number of delays to be used by the model
    
    omega=1.2                      ## Omega of sin(omega*t) of the Duffing oscillator
    dt=0.09*(2*torch.pi/omega)     ## Discretization step adopted by the numerical method
    Amplitude=0.2                  ## Maximum amplitude of the external input signal. The external input signal will be square
                                   ## waves with amplitudes randomly drawn in the interval [-Amplitude Amplitude]
    
    X0=torch.zeros([2])            ## Inital condition. For real data, this would be a distribution, but here we control the 
                                   ## inital condition of the dynamical system
    X0[0]=-0.06
    
    RK=4                           ## RK to be adopted. RK=2 for the chosen discretization step and this system would be unstable 
    N_S=1                          ## External input dimensionality  
    

if Dynamical_System=='LI':
    
    ## Same variables definition as above (but different values)
    sigma_real=torch.tensor(0.5,device=device)  
        
    N_aux=2
    NX_dim=1
    N_noise=2

    N_delays=0

    dt=0.1
    Amplitude=2

    X0=torch.zeros([1])
    
    RK=2
    N_S=1

    
if Dynamical_System=='Rings':
    
    from LI_Definition import *
    
    ## Same variables definition as above (but different values)
    N_aux=1
    NX_dim=1

    N_delays=4

    dt=np.pi/10
    Amplitude=1.25

    X0=torch.zeros([6])

    RK=2
    N_S=2
    
if Dynamical_System=='ASVI':
    
    from LI_Definition import *
    
    ## Same variables definition as above (but different values)
    N_aux=1
    NX_dim=60

    N_delays=0

    dt=1/10
    Amplitude=1

    X0=torch.zeros([60])

    RK=2
    N_S=1


