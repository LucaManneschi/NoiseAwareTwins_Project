import torch
from torch import nn
from torch import optim
import numpy as np

from Functions_Modules import F_Module, G_Module, D_Module
from SDE_Int_Methods import SDE_IntMethods, ODE_IntMethods


## THE CLASS TO DEFINE NEURAL ODE/SDE AND TRAIN IT


class NeuralSDE_BPTT(nn.Module):
    def __init__(
        self,
        F_Ns,
        G_Ns,
        D_Ns,
        N_aux,
        N_ext_aux,
        sigmas_model,
        NX_dim,
        N_S,
        device,
        load_weightsF=[],
        load_weightsD=[],
    ):
        super().__init__()

        self.N_aux = N_aux  ## NUMBER OF AUXILIARY VARIABLES USED
        self.N_ext_aux = N_ext_aux  ## NUMBER OF EXTERNAL AUXILIARY VARIABLES ADOPTED (I would set this to zero, it is an experimental feature)
        self.NX_dim = NX_dim  ## OBSERVABLE DIMENSIONALITY OF THE SYSTEM (OF X, WHAT I CALLED IN THE INTRODUCTION N_X)
        self.N_X = F_Ns[
            -1
        ]  ## FULL DIMENSIONALITY OF THE CONSIDERED VALUES OF X, INCLUDING THE DELAYS
        self.N_S = N_S  ## DIMENSIONALITY OF THE EXTERNAL INPUT SIGNAL

        self.F_Ns = F_Ns  ## NUMBER OF NODES AND LAYERS OF THE DETERMINISTIC F FUNCTION. For instance, F_Ns=[5,200,200,4] corresponds to a
        ## neural network with a 5-d input, one hidden layer with 200 nodes...output layer with 4 nodes

        self.G_Ns = G_Ns  ## NUMBER OF NODES AND LAYERS OF THE STOCHASTIC G FUNCTION.
        self.D_Ns = D_Ns  ## NUMBER OF NODES AND LAYERS OF THE DISCRIMINATOR D.

        self.device = device

        ## INITIALISATION OF F
        self.F = F_Module(F_Ns, N_aux, N_ext_aux, load_weightsF, NX_dim, device).to(
            device
        )

        ## INITIALISATION OF G
        self.G = G_Module(G_Ns, F_Ns, N_aux + N_ext_aux, sigmas_model, device).to(
            device
        )

        ## INITIALISATION OF D
        self.D = D_Module(D_Ns, load_weightsD, device).to(device)

        ## PARAMETERS FOR EFFECT OF THE EXTERNAL VARIABLES (SHOULD BE EMPTY)
        self.W_ext = torch.nn.Parameter(
            torch.ones([N_ext_aux, self.NX_dim], device=device)
        )

        ## SDE=TRUE WILL DEFINE AN SDE, EXPLOITING A STOCHASTIC NUMERICAL METHOD, SDE=FALSE DEFINES AN ODE
        self.SDE = True

    def Initialise_Hyperparameters(
        self, dt, etas, lamda, sigmas_model, Train_Type, RK=4
    ):
        self.RK = RK

        self.dt = dt  ## VALUE OF THE DISCRETIZATION STEP
        self.sigmas_model = sigmas_model  ## BINARY TENSOR DEFINING WHERE THE NOISE IS INTRODUCED ACROSS VARIABLES
        ## Practically, it is a multiplicative factor sigma_model*G*dW,
        ## so setting this to zero on dimension i will suppress the noise contribution on that dimension

        self.Int_Sde = SDE_IntMethods(
            self.F, self.G, self.sigmas_model, self.dt, RK, self.device
        )  ## Definition of the SDE integration method,
        ## where the F and G functions are the neural networks F_Module, G_Module

        self.Int_Ode = ODE_IntMethods(self.F, self.dt, RK, self.device)

        self.etas = etas  ## Learning rates for deterministic (etas[0]), stochastic (etas[1]) and G2F parameters (etas[2])

        self.opt_F = optim.Adam(
            self.F.F.parameters(), lr=etas[0]
        )  ## Optimizer for deterministic F

        if (
            self.N_aux == 0 and self.N_ext_aux == 0
        ):  ## True if auxiliary variables are present
            self.opt_G = optim.Adam(
                self.G.parameters(), lr=etas[1], betas=(0.0, 0.9)
            )  ## Optimizer for stochastic G

        else:
            self.opt_G = optim.Adam(
                self.G.parameters(), lr=etas[1], betas=(0.0, 0.9)
            )  ## Optimizer for stochastic G

            self.opt_G_noise = optim.Adam(
                self.F.G2F.parameters(), lr=etas[2], betas=(0.0, 0.9)
            )  ## Optimizer for G2F parameters

        self.opt_D = optim.Adam(
            self.D.parameters(), lr=etas[1], betas=(0.0, 0.9)
        )  ## Optimizer for the discriminator

        self.lamda = (
            lamda  ## Magnitude of the gradient penalty contribution, usually ten
        )

        self.Train_Type = Train_Type  ## Training type performed by the neural-SDE

    def Reset(
        self, t0, X0
    ):  ## Reset of the activities of the system, i.e. we are starting to compute a new trajectory
        self.t = t0
        self.X = torch.clone(X0)

    def SDE_step(self, Input):  ## Step of the Numerical method
        ## SDE numerical method
        if self.SDE:
            ## Runge-Kutta 4
            if self.RK == 4:
                self.X = self.Int_Sde.RK4(self.X, Input, self.t)

            ## Runge-Kutta 2
            if self.RK == 2:
                self.X = self.Int_Sde.RK2(self.X, Input, self.t)

        ## ODE numerical methods
        else:
            ## Runge-Kutta 4
            if self.RK == 4:
                self.X = self.Int_Ode.RK4(self.X, Input, self.t)
            ## Runge-Kutta 2
            if self.RK == 2:
                self.X = self.Int_Ode.RK2(self.X, Input, self.t)

        self.t = self.t + self.dt

    ## METHOD COMPUTING A TRAJECTORY GIVEN THE Inputs...in this case, gradients are not calculated
    ## and this method should be used for evaluation
    def SDE_Seqforward(self, Inputs, t0):
        with torch.no_grad():
            X = self.forward(Inputs, t0)

        return X

    ## FORWARD PASS, BUT COMPUTING THE GRADIENTS
    def forward(self, Inputs, t0):
        T = Inputs.size()[2]
        batch_size = Inputs.size()[0]
        X = torch.zeros([batch_size, self.N_X, T], device=Inputs.device)

        if self.N_aux == 0 and self.N_ext_aux == 0:
            ## THE ACTIVITIES OF THE SYSTEM ARE FROM N_S ONWARDS...SO WE CAN USE THEM AT TIME T0 FOR INITIALISATION
            self.Reset(t0, Inputs[:, self.N_S :, 0])

        else:
            ## HERE THE AUXILIARY VARIABLES ARE INITIALISED AS 0
            X_reset = torch.concat(
                [
                    Inputs[:, self.N_S :, 0],
                    torch.zeros(
                        [Inputs.size()[0], self.N_aux + self.N_ext_aux],
                        device=Inputs.device,
                    ),
                ],
                1,
            )
            self.Reset(t0, X_reset)

        ## COMPUTATION OF DYNAMICS
        for t in range(T):
            self.SDE_step(Inputs[:, 0 : self.N_S, t])
            X[:, :, t] = torch.clone(self.X[:, 0 : self.N_X])

            if self.N_ext_aux > 0:
                X[:, 0 : self.NX_dim, t] = X[:, 0 : self.NX_dim, t] + torch.matmul(
                    self.X[:, -self.N_ext_aux :], self.F.G2F.W_ext
                )

        return X

    ## METHOD TO TRAIN THE DISCRIMINATOR
    def Train_D(self, Inputs, real_data_original, t0):
        with torch.no_grad():
            ## Generation of fake trajectories by the SDE
            fake_data_original = self.forward(Inputs, t0)

        ## Resetting accumulated gradients in the discriminator
        self.opt_D.zero_grad()

        ## Preparing the data for the discriminator
        real_data_original = real_data_original[:, 0 : self.NX_dim, :]
        fake_data_original = fake_data_original[:, 0 : self.NX_dim, :]

        # External_S=torch.clone(torch.reshape(Inputs[:,0,:],[fake_data_original.size()[0],-1]))
        External_S = torch.clone(
            torch.reshape(
                Inputs[:, 0 : self.N_S, :], [fake_data_original.size()[0], -1]
            )
        )

        fake_data = torch.reshape(
            fake_data_original, [fake_data_original.size()[0], -1]
        )
        fake_data = torch.concat([fake_data, External_S], 1)

        real_data = torch.reshape(real_data_original, [fake_data.size()[0], -1])
        real_data = torch.concat([real_data, External_S], 1)

        ## Computation of discriminator response for fake data
        D_real, _ = self.D(real_data)
        D_real = -D_real.mean()
        D_real.backward()

        ## Computation of discriminator response for real data
        D_fake, _ = self.D(fake_data)
        D_fake = D_fake.mean()
        D_fake.backward()

        ## Gradient penalty
        GP = self.calc_gradient_penalty(real_data, fake_data)
        GP = torch.mean(GP)
        GP.backward()

        ## Computation of loss function for visualization purposes
        D_cost = (D_fake - D_real + GP).detach()
        D_W = (D_real - D_fake).detach()

        ## Optimizer step
        self.opt_D.step()

        return D_cost, D_W, fake_data_original

    def Train_F(self, Inputs, Targets, t0):
        self.opt_F.zero_grad()

        if (self.N_aux + self.N_ext_aux) > 0:
            self.opt_G_noise.zero_grad()

        # T=Targets.size()[2]

        X = self.forward(Inputs, t0)

        mse = torch.mean(
            torch.sum(torch.mean((Targets[:, :, :] - X[:, :, :]) ** 2, 1), 1)
        )

        Err = mse
        Err.backward()

        self.opt_F.step()

        return Err

    def Train_G(self, Inputs, real_data_original, t0, Train_F=False):
        self.opt_G.zero_grad()
        self.opt_D.zero_grad()
        self.opt_F.zero_grad()

        if (self.N_aux + self.N_ext_aux) > 0:
            self.opt_G_noise.zero_grad()

        fake_data_original = self.forward(Inputs, t0)
        fake_data_original = fake_data_original[:, 0 : self.NX_dim, :]
        real_data_original = real_data_original[:, 0 : self.NX_dim, :]

        fake_data = torch.reshape(
            fake_data_original, [fake_data_original.size()[0], -1]
        )
        real_data = torch.reshape(
            real_data_original, [real_data_original.size()[0], -1]
        )
        External_S = torch.clone(
            torch.reshape(Inputs[:, 0 : self.N_S, :], [fake_data.size()[0], -1])
        )

        fake_data = torch.concat([fake_data, External_S], 1)
        real_data = torch.concat([real_data, External_S], 1)

        D_fake, Df_fake = self.D(fake_data)

        G_cost = torch.zeros([np.shape(self.Train_Type)[0]])

        ## DEFINITION OF LOSS FUNCTION (which varies depending on the option selected)
        ## and generator optimizer step
        if self.Train_Type[0] == 1:
            D_fake_E = -D_fake.mean()
            D_fake_E.backward(retain_graph=True)

            G_cost[0] = (-D_fake_E).detach()

        if np.sum(self.Train_Type[1:]) > 0:
            D_real, Df_real = self.D(real_data)

        if self.Train_Type[1] == 1:
            D_mean_E = torch.mean(
                torch.pow(torch.mean(Df_fake, 0) - torch.mean(Df_real, 0), 2),
            )
            D_mean_E.backward(retain_graph=True)
            G_cost[1] = (D_mean_E).detach()

        if self.Train_Type[2] == 1:
            D_var_E = torch.mean(
                torch.pow(
                    torch.sqrt(torch.var(Df_fake, 0))
                    - torch.sqrt(torch.var(Df_real, 0)),
                    2,
                )
            )
            D_var_E.backward(retain_graph=True)
            G_cost[2] = (D_var_E).detach()

        self.opt_G.step()
        self.opt_G_noise.step()
        if Train_F:
            self.opt_F.step()

        return G_cost, fake_data_original

    ## Gradient penalty computation
    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size()[0]

        alpha = torch.rand([batch_size, 1], device=fake_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.requires_grad_()

        disc_interpolates, _ = self.D(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=fake_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        GP = (
            (gradients.reshape([gradients.size()[0], -1]).norm(2, dim=1) - 1) ** 2
        ).mean() * self.lamda

        return GP
