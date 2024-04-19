from hypers import Params
import utils

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import optuna
import numpy as np

import time
import cProfile
import pstats
import argparse
import os
import psutil


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=0, help="")
    parser.add_argument("--samples", type=int, default=0, help="")
    return parser.parse_args()


def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    mask = 1.0 * B

    return mask


def generate_hint(mask, hint_rate):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask

    return hint


def update_D(batch, mask, hint, Z, net_D, net_G, loss, optimizer_D):
    new_X = mask * batch + (1 - mask) * Z

    input_G = torch.cat((new_X, mask), 1).float()

    sample_G = net_G(input_G)
    fake_X = new_X * mask + sample_G * (1 - mask)
    fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
    fake_Y = net_D(fake_input_D)

    loss_D = (loss(fake_Y.float(), mask.float())).mean()

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    return loss_D


def update_G(batch, mask, hint, Z, net_D, net_G, loss, optimizer_G, alpha):
    loss_mse = nn.MSELoss(reduction="none")

    ones = torch.ones_like(batch)

    new_X = mask * batch + (1 - mask) * Z
    input_G = torch.cat((new_X, mask), 1).float()
    sample_G = net_G(input_G)
    fake_X = new_X * mask + sample_G * (1 - mask)

    fake_input_D = torch.cat((fake_X, hint), 1).float()
    fake_Y = net_D(fake_input_D)

    # print(batch, mask, ones.reshape(fake_Y.shape), fake_Y, loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask), (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask)).mean())
    loss_G_entropy = (
        loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1 - mask)
    ).mean()
    loss_G_mse = (loss_mse((sample_G * mask).float(), (batch * mask).float())).mean()

    loss_G = loss_G_entropy + alpha * loss_G_mse

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    return loss_G


def generate_sample(data, mask):
    dim = data.shape[1]
    size = data.shape[0]

    Z = torch.rand((size, dim)) * 0.01
    missing_data_with_noise = mask * data + (1 - mask) * Z
    input_G = torch.cat((missing_data_with_noise, mask), 1).float()

    return net_G(input_G)


def train(
    net_D,
    net_G,
    lr_D,
    lr_G,
    data_iter,
    num_epochs,
    data,
    train_data,
    test_data,
    missing_data,
    alpha,
    mask,
    run,
):
    dim = missing_data.shape[1]
    size = missing_data.shape[0]

    # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
    loss = nn.BCELoss(reduction="none")
    loss_mse = nn.MSELoss(reduction="none")

    loss_D_values = np.zeros(num_epochs)
    loss_G_values = np.zeros(num_epochs)
    loss_MSE_train = np.zeros(num_epochs)
    loss_MSE_test = np.zeros(num_epochs)
    loss_MSE_all = np.zeros(num_epochs)
    loss_MSE_testsplit = np.zeros(num_epochs)

    # for w in net_D.parameters():
    #    nn.init.normal_(w, 0, 0.02)
    # for w in net_G.parameters():
    #    nn.init.normal_(w, 0, 0.02)

    # for w in net_D.parameters():
    #    nn.init.xavier_normal_(w)
    # for w in net_G.parameters():
    #    nn.init.xavier_normal_(w)

    # Initialize weights for net_D
    for name, param in net_D.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    # Initialize weights for net_G
    for name, param in net_G.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    # optimizer_D = torch.optim.SGD(net_D.parameters(), lr = lr_D)
    # optimizer_G = torch.optim.SGD(net_G.parameters(), lr = lr_G)

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch, mask_batch, hint_batch in data_iter:
            batch_size = batch.shape[0]

            # Z = torch.normal(0, 1, size=(batch_size, dim))
            Z = torch.rand((batch_size, dim)) * 0.01
            loss_D = update_D(
                batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_D
            )
            loss_G = update_G(
                batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_G, alpha
            )

            # Z = torch.rand((batch_size, dim)) * 0.01
            # missing_data_with_noise = mask_batch * batch + (1 - mask_batch) * Z
            # input_G = torch.cat((missing_data_with_noise, mask_batch), 1).float()
            # sample_G = net_G(input_G)

            sample_G = generate_sample(batch, mask_batch)

            loss_MSE_train[epoch] = (
                loss_mse(mask_batch * batch, mask_batch * sample_G)
            ).mean()

            loss_MSE_test[epoch] = (
                loss_mse((1 - mask_batch) * batch, (1 - mask_batch) * sample_G)
            ).mean() / (1 - mask_batch).mean()

        if epoch % 100 == 0:
            s = f"{epoch}: loss D={loss_D.detach().numpy(): .3f}  loss G={loss_G.detach().numpy(): .3f}  mse train={loss_MSE_train[epoch]: .4f}  mse test={loss_MSE_test[epoch]: .3f}"
            pbar.clear()
            # logger.info('{}'.format(s))
            pbar.set_description(s)

        data_test = test_data[:][0]
        mask_test = test_data[:][1]

        # Z = torch.rand((len(data_test), dim)) * 0.01
        # missing_data_with_noise = mask_test * data_test + (1 - mask_test) * Z
        # input_G = torch.cat((missing_data_with_noise, mask_test), 1).float()
        # sample_G = net_G(input_G)

        sample_G = generate_sample(data_test, mask_test)

        loss_MSE_testsplit[epoch] = (
            loss_mse((1 - mask_test) * data_test, (1 - mask_test) * sample_G)
        ).mean() / (1 - mask_test).mean()

        # Z = torch.rand((size, dim)) * 0.01
        # missing_data_with_noise = mask * missing_data + (1 - mask) * Z
        # input_G = torch.cat((missing_data_with_noise, mask), 1).float()
        # sample_G = net_G(input_G)

        sample_G = generate_sample(missing_data, mask)

        loss_D_values[epoch] = loss_D.detach().numpy()
        loss_G_values[epoch] = loss_G.detach().numpy()
        loss_MSE_all[epoch] = (
            loss_mse((1 - mask) * data, (1 - mask) * sample_G)
        ).mean() / (1 - mask).mean()

        data_test = test_data[:][0]
        mask_test = test_data[:][1]

        # print("Data:\n", data, scaler.inverse_transform(data), "\nImputed:\n", fake_X, scaler.inverse_transform(fake_X.detach().numpy()))

    data_imputed = missing_data * mask + sample_G * (1 - mask)
    data_imputed = scaler.inverse_transform(data_imputed.detach().numpy())

    utils.create_csv(
        data_imputed,
        folder + "results/" + dataset + f"Imputed_{int(params.miss_rate * 100)}_{run}",
        data_header,
    )
    utils.create_csv(
        loss_D_values,
        folder + f"results/lossD_{int(params.miss_rate * 100)}_{run}",
        "loss D",
    )
    utils.create_csv(
        loss_G_values,
        folder + f"results/lossG_{int(params.miss_rate * 100)}_{run}",
        "loss G",
    )
    utils.create_csv(
        loss_MSE_train,
        folder + f"results/lossMSE_train_{int(params.miss_rate * 100)}_{run}",
        "loss MSE train",
    )

    utils.create_csv(
        loss_MSE_test,
        folder + f"results/lossMSE_test_{int(params.miss_rate * 100)}_{run}",
        "loss MSE test",
    )

    utils.create_csv(
        loss_MSE_all,
        folder + f"results/lossMSE_all_{int(params.miss_rate * 100)}_{run}",
        "loss MSE all",
    )

    utils.create_csv(
        loss_MSE_testsplit,
        folder + f"results/lossMSE_testsplit_{int(params.miss_rate * 100)}_{run}",
        "loss MSE testsplit",
    )


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def train_v2(
    net_D,
    net_G,
    lr_D,
    lr_G,
    data_iter,
    num_epochs,
    batch_size,
    train_size,
    complete_train,
    complete_test,
    combined_dataset_train,
    combined_dataset_test,
    missing_data,
    alpha,
    mask,
    run,
):
    cpu = []
    ram = []
    ram_percentage = []

    dim = missing_data.shape[1]
    size = missing_data.shape[0]

    missing_test = combined_dataset_test[:][0]
    mask_test = combined_dataset_test[:][1]

    # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
    loss = nn.BCELoss(reduction="none")
    loss_mse = nn.MSELoss(reduction="none")

    loss_D_values = np.zeros(num_epochs)
    loss_G_values = np.zeros(num_epochs)
    loss_MSE_train = np.zeros(num_epochs)
    loss_MSE_test = np.zeros(num_epochs)
    loss_MSE_train_testsplit = np.zeros(num_epochs)
    loss_MSE_test_testsplit = np.zeros(num_epochs)

    # for w in net_D.parameters():
    #    nn.init.normal_(w, 0, 0.02)
    # for w in net_G.parameters():
    #    nn.init.normal_(w, 0, 0.02)

    # for w in net_D.parameters():
    #    nn.init.xavier_normal_(w)
    # for w in net_G.parameters():
    #    nn.init.xavier_normal_(w)

    # Initialize weights for net_D
    for name, param in net_D.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    # Initialize weights for net_G
    for name, param in net_G.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:

        mb_idx = sample_idx(train_size, batch_size)

        batch = torch.stack([combined_dataset_train[idx][0] for idx in mb_idx])
        mask_batch = torch.stack([combined_dataset_train[idx][1] for idx in mb_idx])
        hint_batch = torch.stack([combined_dataset_train[idx][2] for idx in mb_idx])

        Z = torch.rand((batch_size, dim)) * 0.01
        loss_D = update_D(
            batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_D
        )
        loss_G = update_G(
            batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_G, alpha
        )

        sample_G = generate_sample(batch, mask_batch)

        loss_MSE_train[epoch] = (
            loss_mse(mask_batch * batch, mask_batch * sample_G)
        ).mean()

        loss_MSE_test[epoch] = (
            loss_mse((1 - mask_batch) * batch, (1 - mask_batch) * sample_G)
        ).mean() / (1 - mask_batch).mean()

        if epoch % 100 == 0:
            s = f"{epoch}: loss D={loss_D.detach().numpy(): .3f}  loss G={loss_G.detach().numpy(): .3f}  rmse train={np.sqrt(loss_MSE_train[epoch]): .4f}  rmse test={np.sqrt(loss_MSE_test[epoch]): .3f}"
            pbar.clear()
            # logger.info('{}'.format(s))
            pbar.set_description(s)

            cpu.append(psutil.cpu_percent())
            ram.append(psutil.virtual_memory()[3] / 1000000000)
            ram_percentage.append(psutil.virtual_memory()[2])

        sample_G = generate_sample(missing_test, mask_test)
        loss_MSE_train_testsplit[epoch] = (
            loss_mse(mask_test * missing_test, mask_test * sample_G)
        ).mean()

        sample_G = generate_sample(missing_test, mask_test)
        loss_MSE_test_testsplit[epoch] = (
            loss_mse((1 - mask_test) * complete_test, (1 - mask_test) * sample_G)
        ).mean() / (1 - mask_test).mean()

        loss_D_values[epoch] = loss_D.detach().numpy()
        loss_G_values[epoch] = loss_G.detach().numpy()

        # print("Data:\n", data, scaler.inverse_transform(data), "\nImputed:\n", fake_X, scaler.inverse_transform(fake_X.detach().numpy()))

    data_test_imputed = missing_test * mask_test + sample_G * (1 - mask_test)
    data_test_imputed = scaler.inverse_transform(data_test_imputed.detach().numpy())

    sample_G = generate_sample(missing_data, mask)
    data_train_imputed = missing_data * mask + sample_G * (1 - mask)
    data_train_imputed = scaler.inverse_transform(data_train_imputed.detach().numpy())

    utils.create_csv(
        data_train_imputed,
        folder
        + "results/"
        + dataset
        + f"Imputed_{int(params.miss_rate * 100)}_{train_size}_{run}",
        data_train_header,
    )
    utils.create_csv(
        data_test_imputed,
        folder
        + "results/"
        + dataset
        + f"Imputed_test_{int(params.miss_rate * 100)}_{train_size}_{run}",
        data_train_header,
    )
    utils.create_csv(
        loss_D_values,
        folder + f"results/lossD_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss D",
    )
    utils.create_csv(
        loss_G_values,
        folder + f"results/lossG_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss G",
    )
    utils.create_csv(
        loss_MSE_train,
        folder
        + f"results/lossMSE_train_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss MSE train",
    )

    utils.create_csv(
        loss_MSE_test,
        folder
        + f"results/lossMSE_test_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss MSE test",
    )

    utils.create_csv(
        loss_MSE_test_testsplit,
        folder
        + f"results/lossMSE_test_testsplit_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss MSE all",
    )

    utils.create_csv(
        loss_MSE_train_testsplit,
        folder
        + f"results/lossMSE_train_testsplit_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "loss MSE testsplit",
    )

    utils.create_csv(
        cpu,
        folder + f"results/cpu_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "cpu usage",
    )

    utils.create_csv(
        ram,
        folder + f"results/ram_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "ram usage",
    )

    utils.create_csv(
        ram_percentage,
        folder
        + f"results/ram_percentage_{int(params.miss_rate * 100)}_{train_size}_{run}",
        "ram percentage usage",
    )


if __name__ == "__main__":
    start_time = time.time()
    with cProfile.Profile() as profile:

        # train_samples = [455, 409, 364, 318, 273, 227, 204, 182, 159, 136]
        test_samples = 114  # breast size

        # train_samples = [
        #     3680,
        #     3312,
        #     2944,
        #     2576,
        #     2208,
        #     1840,
        #     1656,
        #     1472,
        #     1288,
        #     1104,
        #     920,
        #     736,
        #     552,
        #     368,
        # ]
        # test_samples = 921  # spam size

        # train_samples = [
        #     24000,
        #     21600,
        #     19200,
        #     16800,
        #     14400,
        #     12000,
        #     10800,
        #     9600,
        #     8400,
        #     7200,
        #     6000,
        #     4800,
        #     3600,
        #     2400,
        # ]
        # test_samples = 6000  # credit size

        args = init_arg()

        run = args.run
        samples = args.samples

        dataset = "breast"
        folder = "/home/leandrosobral/LeandroSobralThesis/" + dataset + "/"

        params = Params.read_hyperparameters("parameters.json")
        loss_MSE_train_final = np.zeros(params.num_runs)
        loss_MSE_test_final = np.zeros(params.num_runs)
        run_time = np.zeros(params.num_runs)

        df_data_train = pd.read_csv(
            folder + dataset + f"_{int(params.miss_rate * 100)}_{samples}.csv"
        )
        # features = list(df_data.columns)
        data_train = df_data_train.values
        data_train_header = df_data_train.columns.tolist()

        df_data_test = pd.read_csv(
            folder + dataset + f"_test_{int(params.miss_rate * 100)}_{test_samples}.csv"
        )
        data_test = df_data_test.values
        data_test_header = df_data_test.columns.tolist()

        df_missing_train = pd.read_csv(
            f"{folder}{dataset}Missing_{int(params.miss_rate * 100)}_{samples}.csv"
        )
        missing_train = df_missing_train.values

        df_missing_test = pd.read_csv(
            f"{folder}{dataset}Missing_test_{int(params.miss_rate * 100)}_{test_samples}.csv"
        )
        missing_test = df_missing_test.values

        mask_train = np.where(np.isnan(missing_train), 0.0, 1.0)
        missing_data = np.where(mask_train, missing_train, 0.0)
        hint = generate_hint(mask_train, params.hint_rate)

        mask_test = np.where(np.isnan(missing_test), 0.0, 1.0)
        missing_test = np.where(mask_test, missing_test, 0.0)

        range_scaler = (0, 1)
        scaler = MinMaxScaler(feature_range=range_scaler)
        missing_data = scaler.fit_transform(missing_data)
        data_train = scaler.transform(data_train)
        # mask = scaler.transform(mask)
        # hint = scaler.transform(hint)
        data_test = scaler.transform(data_test)
        missing_test = scaler.transform(missing_test)

        dim = missing_data.shape[1]
        train_size = missing_data.shape[0]

        missing_data = torch.from_numpy(missing_data)
        mask_train = torch.from_numpy(mask_train)
        hint = torch.from_numpy(hint)

        data_test = torch.from_numpy(data_test)
        missing_test = torch.from_numpy(missing_test)
        mask_test = torch.from_numpy(mask_test)

        combined_dataset_train = torch.utils.data.TensorDataset(
            missing_data, mask_train, hint
        )
        combined_dataset_test = torch.utils.data.TensorDataset(missing_test, mask_test)

        # train_size = len(missing_train)
        # train_data, test_data = torch.utils.data.random_split(
        #    combined_dataset, [train_size, size - train_size]
        # )

        h_dim1 = dim
        h_dim2 = dim

        print("\n\nStarting run number", run + 1, "out of", params.num_runs, "\n")

        data_iter = torch.utils.data.DataLoader(
            missing_train, params.batch_size, pin_memory=True, num_workers=8
        )

        net_G = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        net_D = nn.Sequential(
            nn.Linear(dim * 2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, dim),
            nn.Sigmoid(),
        )

        train_v2(
            net_D,
            net_G,
            params.lr_D,
            params.lr_G,
            data_iter,
            params.num_epochs,
            params.batch_size,
            train_size,
            data_train,
            data_test,
            combined_dataset_train,
            combined_dataset_test,
            missing_data,
            params.alpha,
            mask_train,
            run,
        )

        run_time = []
        run_time.append(time.time() - start_time)
        file_path = (
            folder + f"results/run_time_{int(params.miss_rate * 100)}_{samples}.csv"
        )
        if run == 0:
            df_run_time = pd.DataFrame(run_time)
            df_run_time.to_csv(file_path, index=False)
        else:
            if os.path.exists(file_path):
                with open(file_path, "a") as myfile:
                    myfile.write(str(run_time[0]) + "\n")

            else:
                df_run_time = pd.DataFrame(run_time)
                df_run_time.to_csv(file_path, index=False)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    results.dump_stats("results.prof")

    time_delta = []
    time_delta.append(time.time() - start_time)
    print("--- %s seconds ---" % (time_delta[0]))

    utils.create_csv(
        time_delta,
        folder
        + "results/"
        + f"total_run_time_{int(params.miss_rate * 100)}_{train_size}",
        "Total run time (s)",
    )
