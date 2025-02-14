import torch
from math import sqrt


## STOCHASTIC RK2 AND RK4
class SDE_IntMethods:
    def __init__(self, f, g, sigmas, dt, RK=4, device="cpu"):
        self.dt = dt
        self.root_dt = sqrt(dt)
        self.root_2_o_dt = sqrt(2.0 / dt)
        self.root_1_o_dt = sqrt(1.0 / dt)

        self.f = f
        self.g = g
        self.sigmas = sigmas
        self.N_noise = sigmas.size()[1]

        self.RK = RK
        self.device = device

    def RK4(self, X, I, t):
        Tilde = (self.sigmas.unsqueeze(0)) * torch.randn(
            [X.size()[0], X.size()[1], self.N_noise], device=self.sigmas.device
        )

        k1 = self.f(X, t, I[:, :])
        l1 = self.g(X, t, I[:, :])

        k1_ = k1 + self.root_2_o_dt * torch.einsum("ijk,ijk->ij", l1, Tilde)

        k2 = self.f(X + k1_ * self.dt / 2, t + 0.5 * self.dt, I[:, :])
        l2 = self.g(X + k1_ * self.dt / 2, t + 0.5 * self.dt, I[:, :])
        k2_ = k2 + self.root_2_o_dt * torch.einsum("ijk,ijk->ij", l2, Tilde)

        k3 = self.f(X + k2_ * self.dt / 2, t + self.dt / 2, I[:, :])
        l3 = self.g(X + k2_ * self.dt / 2, t + self.dt / 2, I[:, :])
        k3_ = k3 + self.root_1_o_dt * torch.einsum("ijk,ijk->ij", l3, Tilde)

        k4 = self.f(X + k3_ * self.dt, t + self.dt, I[:, :])
        l4 = self.g(X + k3_ * self.dt, t + self.dt, I[:, :])

        x_new = (
            X
            + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
            + sqrt(self.dt) * torch.einsum("ijk,ijk->ij", (l1 + 2 * l2 + 2 * l3 + l4)/6, Tilde)
        )

        return x_new

    def RK2(self, X, I, t):
        Tilde = (self.sigmas.unsqueeze(0)) * torch.randn(
            [X.size()[0], X.size()[1], self.N_noise], device=X.device
        )

        k1 = self.f(X, t, I[:, :])
        l1 = self.g(X, t, I[:, :])

        k1_ = k1 + sqrt(1 / self.dt) * torch.einsum("ijk,ijk->ij", l1, Tilde)

        k2 = self.f(X + k1_ * self.dt, t + self.dt, I[:, :])
        l2 = self.g(X + k1_ * self.dt, t + self.dt, I[:, :])

        x_new = (
            X
            + 0.5 * (k1 + k2) * self.dt
            + sqrt(self.dt) * torch.einsum("ijk,ijk->ij", 0.5 * (l1 + l2), Tilde)
        )

        return x_new

    def Compute_Dynamics(self, Input, x0, t0):
        T = Input.size()[2]
        batch_size = x0.size()[0]
        N = x0.size()[1]

        X = torch.zeros([batch_size, N, T]).to(Input.device)
        X[:, :, 0] = x0

        t = t0

        if self.RK == 4:
            for n in range(1, T):
                I = Input[:, :, n]

                X[:, :, n] = self.RK4(X[:, :, n - 1], I, t)

        if self.RK == 2:
            for n in range(1, T):
                I = Input[:, :, n]

                X[:, :, n] = self.RK2(X[:, :, n - 1], I, t)

        return X


## Standard RK2 and RK4


class ODE_IntMethods:
    def __init__(self, f, dt, RK=4, device="cpu"):
        self.dt = dt
        self.f = f
        self.device = device

        self.RK = RK

    def RK4(self, X, I, t):
        k1 = self.f(X, t, I[:, :])

        k2 = self.f(X + k1 * self.dt / 2, t + self.dt / 2, I[:, :])

        k3 = self.f(X + k2 * self.dt / 2, t + self.dt / 2, I[:, :])

        k4 = self.f(X + k3 * self.dt, t + self.dt, I[:, :])

        x_new = X + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * self.dt

        return x_new

    def RK2(self, X, I, t):
        k1 = self.f(X, t, I[:, :])

        k2 = self.f(X + k1 * self.dt, t + self.dt, I[:, :])

        x_new = X + 0.5 * (k1 + k2) * self.dt

        return x_new

    def Compute_Dynamics(self, Input, x0, t0):
        T = Input.size()[2]
        batch_size = x0.size()[0]
        N = x0.size()[1]

        X = torch.zeros([batch_size, N, T]).to(self.device)
        X[:, :, 0] = x0

        t = t0

        if self.RK == 4:
            for n in range(1, T):
                I = Input[:, :, n]

                X[:, :, n] = self.RK4(X[:, :, n - 1], I, t)

        if self.RK == 2:
            for n in range(1, T):
                I = Input[:, :, n]

                X[:, :, n] = self.RK2(X[:, :, n - 1], I, t)

        return X


def AutoCov(S1, S2):
    N_cases = S1.size()[0]
    batch_size = S1.size()[1]
    N_percase = int(batch_size / N_cases)
    N_dim = S1.size()[2]
    T = S1.size()[3]

    ACov1 = torch.zeros([N_cases, N_dim, T, T], device=S1.device)
    ACov2 = torch.zeros([N_cases, N_dim, T, T], device=S1.device)
    Means1 = torch.zeros([N_cases, N_dim, T], device=S1.device)
    Means2 = torch.zeros([N_cases, N_dim, T], device=S1.device)
    Var1 = torch.zeros([N_cases, N_dim, T], device=S1.device)
    Var2 = torch.zeros([N_cases, N_dim, T], device=S1.device)

    for n in range(N_cases):
        for m in range(N_dim):
            s1 = S1[n, :, m, :]
            s2 = S2[n, :, m, :]

            Means1[n, m, :] = torch.mean(s1, 0)
            Means2[n, m, :] = torch.mean(s2, 0)

            s1_m = (s1 - torch.mean(s1, 0).unsqueeze(0)).unsqueeze(2)
            s1_m_ = (s1 - torch.mean(s1, 0).unsqueeze(0)).unsqueeze(1)

            s2_m = (s2 - torch.mean(s2, 0).unsqueeze(0)).unsqueeze(2)
            s2_m_ = (s2 - torch.mean(s2, 0).unsqueeze(0)).unsqueeze(1)

            if s1.size()[0] > 1 and s2.size()[0] > 1:
                Var1[n, m, :] = torch.sqrt(torch.var(s1, 0))
                Var2[n, m, :] = torch.sqrt(torch.var(s2, 0))

            ACov1[n, m, :, :] = torch.mean(s1_m * s1_m_, 0) / (
                Var1[n, m, :].unsqueeze(1) * Var1[n, m, :].unsqueeze(0) + 0.001
            )
            ACov2[n, m, :, :] = torch.mean(s2_m * s2_m_, 0) / (
                Var2[n, m, :].unsqueeze(1) * Var2[n, m, :].unsqueeze(0) + 0.001
            )

    Errors = torch.zeros([3])

    MV1 = torch.mean(Var1, 2)
    MV2 = torch.mean(Var2, 2)

    ws = (MV1 + MV2) / (
        torch.sum(MV1, 0).unsqueeze(0) + torch.sum(MV2, 0).unsqueeze(0) + 0.001
    )

    Errors[0] = torch.mean(torch.sqrt(torch.pow(Means1 - Means2, 2)))
    Errors[1] = torch.mean(
        torch.sum(torch.sqrt(torch.pow(Var1 - Var2, 2)) * ws.unsqueeze(2), 0)
    )
    Errors[2] = torch.mean(
        torch.sum(torch.pow(ACov1 - ACov2, 2) * ws.unsqueeze(2).unsqueeze(3), 0)
    )

    return ACov1, ACov2, Means1, Means2, Var1, Var2, Errors
