import torch


def get_params(Dynamical_System = 'DO'):
    if Dynamical_System=='DO':
        omega = 1.2

        params = {
        'sigma_real':0.05,
        ## Standard deviation of the noise present in the system (it will actually vary
        ## following a function G, but this scales the total level of noise)
        #sigma_real=torch.tensor(0.0,device=device)

        'N_aux':2,                        ## Number of auxiliary variables of the analytical system (not the model)

        'NX_dim':2,                       ## Observable dimensionality of the Duffing oscillator (x and v)
        'N_noise':2,                      ## Noise Dimensionality

        'N_delays':1,                     ## Number of delays to be used by the model
        #N_delays=0                       ## Number of delays to be used by the model

        'omega':omega,                      ## Omega of sin(omega*t) of the Duffing oscillator
        'dt':0.09*(2*torch.pi/omega),     ## Discretization step adopted by the numerical method
        'Amplitude':0.2,                    ## Maximum amplitude of the external input signal. The external input signal will be square
                                                ## waves with amplitudes randomly drawn in the interval [-Amplitude Amplitude]

        'X0':torch.Tensor([0.0, -0.06]),        ## Inital condition. For real data, this would be a distribution, but here we control the
                                                ## inital condition of the dynamical system

        'RK':4,                           ## RK to be adopted. RK=2 for the chosen discretization step and this system would be unstable
        'N_S':1                          ## External input dimensionality
        }

    if Dynamical_System=='LI':

        params = {
        'sigma_real':0.0,
        ## Standard deviation of the noise present in the system (it will actually vary
        ## following a function G, but this scales the total level of noise)
        #sigma_real=torch.tensor(0.0,device=device)

        'N_aux':2,                        ## Number of auxiliary variables of the analytical system (not the model)

        'NX_dim':1,                       ## Observable dimensionality of the Duffing oscillator (x and v)
        'N_noise':2,                      ## Noise Dimensionality

        'N_delays':0,                     ## Number of delays to be used by the model

        'dt':0.1,               ## Discretization step adopted by the numerical method
        'Amplitude':2.0,                  ## Maximum amplitude of the external input signal. The external input signal will be square
                                    ## waves with amplitudes randomly drawn in the interval [-Amplitude Amplitude]

        'X0':torch.Tensor([0.0]),
                    ## Inital condition. For real data, this would be a distribution, but here we control the
                                    ## inital condition of the dynamical system

        'RK':2,                           ## RK to be adopted. RK=2 for the chosen discretization step and this system would be unstable
        'N_S':1                          ## External input dimensionality
        }
    
    
    return params
""" 
if Dynamical_System=='LI':

    ## Same variables definition as above (but different values)
    #sigma_real=torch.tensor(0.5,device=device)
    #sigma_real=torch.tensor(0.1,device=device)
    sigma_real=0.

    N_aux=2
    NX_dim=1
    N_noise=2

    N_delays=0

    dt=0.1
    #dt=0.01
    Amplitude=2

    X0=torch.zeros([1])

    RK=2
    N_S=1


if Dynamical_System=='Rings':

    from LI_Definition import *

    ## Same variables definition as above (but different values)
    N_aux=1
    NX_dim=1
    N_noise=2 ## This actually could be deleted, since the data are not computationally "generated", but are experimental

    N_delays=4

    dt=np.pi/10
    Amplitude=1.25

    X0=torch.zeros([6])

    RK=2
    N_S=2



if Dynamical_System=='ADNI_UNIMODAL':
    
    ## Same variables definition as above (but different values)
    sigma_real=0.02
        
    N_aux=2
    NX_dim=1
    N_noise=2

    N_delays=0

    dt=0.1
    Amplitude=2

    X0=torch.zeros([1])
    
    RK=2
    N_S=0

    

if Dynamical_System=='ADNI_MULTIMODAL':
    
    sigma_real=0.02
        
    NX_dim=3
    N_noise=2

    N_delays=0

    dt=0.1
    Amplitude=2

    X0=torch.zeros([2])
    
    RK=2
    N_S=1
    

     """




