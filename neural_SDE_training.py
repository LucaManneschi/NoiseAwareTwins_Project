import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from SDE_Int_Methods import AutoCov


def train_NSDE_phase1(
    N_SDE,
    N_ST1,
    Input,
    Target,
    Input_val,
    Target_val,
    Input_te,
    Target_te,
    T_horizon,
    params,
    device,
    batch_size,
    t0,
    print_steps=1000,
):
    N_SDE.SDE = False  
    ## Setting it to False means that we will use a deterministic numerical method for convenience of
    ## computation. In this phase, only the deterministic part is trained.

    N_datasets = len(Input)

    Err = torch.zeros([N_ST1])

    print("First Phase of Training")
    Best_Error = torch.tensor(1)

    Saving = False
    Plot_Res = True

    ERR1_val = torch.zeros([N_ST1, 1, 3])
    ERR1_te = torch.zeros([N_ST1, 1, 3])

    RK = params["RK"]
    N_delays = params["N_delays"]
    Dynamical_System = params["Dynamical_System"]

    data_type = 0
    S = torch.zeros([batch_size, Input[data_type].size()[1], T_horizon]).to(device)
    Tg = torch.zeros([batch_size, Target[data_type].size()[1], T_horizon]).to(device)

    for n in tqdm.trange(N_ST1):
        if n >= 0:
            data_type = np.random.randint(0, N_datasets)

            rand_ind = np.random.randint(0, Input[data_type].size()[0], (batch_size,))
            rand_t = np.random.randint(
                0, Input[data_type].size()[2] - T_horizon, (batch_size,)
            )

            S = torch.zeros([batch_size, Input[data_type].size()[1], T_horizon]).to(
                device
            )
            Tg = torch.zeros([batch_size, Target[data_type].size()[1], T_horizon]).to(
                device
            )

            for k in range(batch_size):
                S[k, :, :] = Input[data_type][
                    rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)
                ]
                Tg[k, :, :] = Target[data_type][
                    rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)
                ]

            mse = N_SDE.Train_F(
                S, Tg, torch.tensor(rand_t * params["dt"], device=device)
            )
            D_cost, D_W, fake_data = N_SDE.Train_D(
                S, Tg, torch.tensor(rand_t * params["dt"], device=device)
            )

            Err[n] = mse.detach()

        if n % print_steps == print_steps - 1:
            ## Test Sets
            ind_data = 0
            if Dynamical_System == "Rings":
                ind_data = 1

            Generated_te = torch.zeros_like(Target_te[ind_data], device=device)
            print(Input_te[0].device)
            for m in range(Input_te[ind_data].size()[0]):
                X_generated = N_SDE.SDE_Seqforward(Input_te[ind_data][m, :, :, :], t0)

                Generated_te[m, :, :, :] = X_generated

            _, _, _, _, _, _, Errors_te = AutoCov(
                Generated_te.to("cpu"), Target_te[ind_data].to("cpu")
            )

            ## Validation Sets

            Generated_val = torch.zeros_like(Target_val[ind_data], device=device)

            for m in range(Input_val[ind_data].size()[0]):
                X_generated = N_SDE.SDE_Seqforward(Input_val[ind_data][m, :, :, :], t0)

                Generated_val[m, :, :, :] = X_generated

            _, _, _, _, _, _, Errors_val = AutoCov(Generated_val, Target_val[ind_data])

            Errors_ = Errors_val[0].detach().to("cpu").unsqueeze(0)

            Best_Error_new = torch.min(
                torch.concat([Best_Error.unsqueeze(0), Errors_], 0)
            )

            ## Saving errors

            ERR1_val[n, 0, :] = Errors_val.detach().to("cpu")
            ERR1_te[n, 0, :] = Errors_te.detach().to("cpu")

            if Best_Error_new < Best_Error and Plot_Res:
                print(
                    "New Plot, Old error: ", Best_Error, "New Error: ", Best_Error_new
                )

                Nplots_x = 3
                Nplots_y = 2
                T_plots = 200
                choice = np.int32(
                    np.linspace(
                        0, Target_val[ind_data].size()[0] - 1, Nplots_x * Nplots_y
                    )
                )

                fig, axs = plt.subplots(Nplots_y, Nplots_x, figsize=(30, 20))

                for i in range(Nplots_y):
                    for j in range(Nplots_x):
                        for k in range(Target_val[ind_data].size()[1]):
                            axs[i, j].plot(
                                Target_val[ind_data][
                                    choice[i * Nplots_x + j], k, 0, 0:T_plots
                                ].to("cpu"),
                                "black",
                                alpha=0.1,
                            )
                            axs[i, j].plot(
                                Generated_val[
                                    choice[i * Nplots_x + j], k, 0, 0:T_plots
                                ].to("cpu"),
                                "red",
                                linewidth=2,
                            )

                plt.show()

            if Best_Error_new < Best_Error and Saving:
                print("Saving... New Error: ", Best_Error_new)
                title_start = (
                    "Parameters_SDE_"
                    + str(RK)
                    + "_Rings_NDelaysCopy_"
                    + str(N_delays)
                    + "T_horizon_"
                    + str(T_horizon)
                    + "_1fInput"
                )

                torch.save(N_SDE, title_start + ".pt")

            Best_Error = torch.clone(Best_Error_new)

            print(n, " Errors on distributions", "Data:", Errors_val, Errors_te)
    return ERR1_val, Best_Error


def train_NSDE_phase2(
    N_SDE,
    N_ST,
    D_step,
    Input,
    Target,
    Input_val,
    Target_val,
    Input_te,
    Target_te,
    T_horizon,
    params,
    device,
    batch_size,
    t0,
    etas,
    Train_F2,
    Errors_val1,
    Best_err_av,
    print_steps=200,
):
    N_datasets = len(Input)
    RK = params["RK"]
    N_delays = params["N_delays"]
    Dynamical_System = params["Dynamical_System"]

    Plot_Res = True

    Saving = False
    Loading = False

    ERR2_val = torch.zeros([N_ST, 1, 3])
    ERR2_te = torch.zeros([N_ST, 1, 3])

    if Loading:
        ## Here one can load a pytorch model
        N_SDE = torch.load(
            "Parameters_SDE_2_Rings_NDelaysCopy_4T_horizon_50_1fInput.pt"
        )

    print("Second Phase of Training")

    N_SDE.opt_F.param_groups[0]["lr"] = etas[1]

    N_SDE.SDE = True

    Train_Type = [0, 1, 1]
    N_SDE.Train_Type = Train_Type

    N_aux_model = params["N_aux_model"]

    sigmas_model = torch.zeros(
        [N_aux_model + params["NX_dim"] * (params["N_delays"] + 1), 2], device=device
    )

    if N_aux_model > 0:
        sigmas_model[-N_aux_model:, :] = 1

    if N_aux_model == 0:
        sigmas_model[-params["NX_dim"] :, :] = 1

    print("Noise Variance Binary Mask:", sigmas_model.T)

    N_SDE.Int_Sde.sigmas = sigmas_model

    Best_Error = torch.tensor(1.0)  ## Value to surpass

    for n in tqdm.trange(N_ST):
        for m in range(D_step):
            data_type = np.random.randint(0, N_datasets)

            rand_ind = np.random.randint(0, Input[data_type].size()[0], (batch_size,))
            rand_t = np.random.randint(
                0, Input[data_type].size()[2] - T_horizon, (batch_size,)
            )

            S = torch.zeros([batch_size, Input[data_type].size()[1], T_horizon]).to(
                device
            )
            Tg = torch.zeros([batch_size, Target[data_type].size()[1], T_horizon]).to(
                device
            )

            for k in range(batch_size):
                S[k, :, :] = torch.clone(
                    Input[data_type][
                        rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)
                    ]
                )
                Tg[k, :, :] = torch.clone(
                    Target[data_type][
                        rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)
                    ]
                )

            D_cost, D_W, fake_data = N_SDE.Train_D(
                S, Tg, torch.tensor(rand_t * params["dt"], device=device)
            )

        data_type = np.random.randint(0, N_datasets)

        rand_ind = np.random.randint(0, Input[data_type].size()[0], (batch_size,))
        rand_t = np.random.randint(
            0, Input[data_type].size()[2] - T_horizon, (batch_size,)
        )

        S = torch.zeros([batch_size, Input[data_type].size()[1], T_horizon]).to(device)
        Tg = torch.zeros([batch_size, Target[data_type].size()[1], T_horizon]).to(
            device
        )

        for k in range(batch_size):
            S[k, :, :] = torch.clone(
                Input[data_type][rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)]
            )
            Tg[k, :, :] = torch.clone(
                Target[data_type][rand_ind[k], :, rand_t[k] + np.arange(0, T_horizon)]
            )

        G_cost, fake_data = N_SDE.Train_G(
            S, Tg, torch.tensor(rand_t * params["dt"], device=device)
        )

        if Train_F2:
            mse = N_SDE.Train_F(
                S, Tg, torch.tensor(rand_t * params["dt"], device=device)
            )

        if n % print_steps == print_steps - 1:
            ## Test Sets

            ind_data = 0
            if Dynamical_System == "Rings":
                ind_data = 1

            Generated_te = torch.zeros_like(Target_te[ind_data], device=device)

            for m in range(Input_te[ind_data].size()[0]):
                X_generated = N_SDE.SDE_Seqforward(Input_te[ind_data][m, :, :, :], t0)

                Generated_te[m, :, :, :] = X_generated

            _, _, _, _, _, _, Errors_te = AutoCov(
                Generated_te.to("cpu"), Target_te[ind_data].to("cpu")
            )

            ## Validation Sets

            Generated_val = torch.zeros_like(Target_val[ind_data], device=device)

            for m in range(Input_val[ind_data].size()[0]):
                X_generated = N_SDE.SDE_Seqforward(Input_val[ind_data][m, :, :, :], t0)

                Generated_val[m, :, :, :] = X_generated

            _, _, _, _, _, _, Errors_val = AutoCov(Generated_val, Target_val[ind_data])

            Errors_ = Errors_val[0].detach().to("cpu").unsqueeze(0)

            Best_Error_new = torch.min(
                torch.concat([Best_Error.unsqueeze(0), Errors_], 0)
            )

            ## Saving errors

            ERR2_val[n, 0, :] = Errors_val.detach().to("cpu")
            ERR2_te[n, 0, :] = Errors_te.detach().to("cpu")

            Errors_ = torch.sum(Errors_val).detach().to("cpu").unsqueeze(0)

            Best_Error_new = torch.min(
                torch.concat([Best_Error.unsqueeze(0), Errors_], 0)
            )

            # if Best_Error_new<Best_Error and Plot_Res:
            if Plot_Res:
                print(
                    "New Plot, Old error: ", Best_Error, "New Error: ", Best_Error_new
                )

                choice = np.int32(np.linspace(0, Target_val[ind_data].size()[0] - 1, 6))

                Nplots_x = 3
                Nplots_y = 2
                T_plots = 200
                fig, axs = plt.subplots(Nplots_y, Nplots_x, figsize=(30, 20))

                for i in range(Nplots_y):
                    for j in range(Nplots_x):
                        for k in range(Target_val[ind_data].size()[1]):
                            axs[i, j].plot(
                                Target_val[ind_data][
                                    choice[i * Nplots_x + j], k, 0, 0:T_plots
                                ].to("cpu"),
                                "black",
                                alpha=0.1,
                            )
                            axs[i, j].plot(
                                Generated_val[
                                    choice[i * Nplots_x + j], k, 0, 0:T_plots
                                ].to("cpu"),
                                "red",
                                alpha=0.1,
                            )

                plt.show()

            if (
                Best_Error_new < Best_Error
                and Saving
                and Errors_val1[0] < Best_err_av + 0.05 * Best_err_av
            ):
                print("Saving... New Error: ", Best_Error_new)
                title_start = (
                    Dynamical_System + "_Parameters_SDE_"
                    + str(RK)
                    + "_NDelays_"
                    + str(N_delays)
                    + "T_horizon_"
                    + str(T_horizon)
                    + "_"
                    + str(Train_Type)
                    + "_V2_1f"
                )

                torch.save(N_SDE, title_start + ".pt")

            Best_Error = torch.clone(Best_Error_new)

            print(n, " Errors on distributions", "Data:", Errors_val, Errors_te)
