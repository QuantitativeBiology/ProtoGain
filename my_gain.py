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


start_time = time.time()


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

    data_test = test_data[:][0]
    mask_test = test_data[:][1]

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

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:

        mb_idx = sample_idx(train_size, batch_size)

        batch = torch.stack([train_data[idx][0] for idx in mb_idx])
        mask_batch = torch.stack([train_data[idx][1] for idx in mb_idx])
        hint_batch = torch.stack([train_data[idx][2] for idx in mb_idx])

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

        sample_G = generate_sample(data_test, mask_test)
        loss_MSE_testsplit[epoch] = (
            loss_mse(mask_test * data_test, mask_test * sample_G)
        ).mean()

        sample_G = generate_sample(missing_data, mask)
        loss_MSE_all[epoch] = (
            loss_mse((1 - mask) * data, (1 - mask) * sample_G)
        ).mean() / (1 - mask).mean()

        loss_D_values[epoch] = loss_D.detach().numpy()
        loss_G_values[epoch] = loss_G.detach().numpy()

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


if __name__ == "__main__":
    with cProfile.Profile() as profile:

        data_samples = [
            455,
            409,
            364,
            318,
            273,
            227,
            204,
            182,
            159,
            136,
            113,
            91,
            68,
            45,
        ]  # breast size
        dataset = "breast"
        folder = "~/LeandroSobralThesis/" + dataset + "/"

        for samples in data_samples:

            params = Params.read_hyperparameters("parameters.json")
            loss_MSE_train_final = np.zeros(params.num_runs)
            loss_MSE_test_final = np.zeros(params.num_runs)
            run_time = np.zeros(params.num_runs)

            df_data = pd.read_csv(folder + dataset + ".csv")
            # features = list(df_data.columns)
            data = df_data.values
            data_header = df_data.columns.tolist()

            df_missing = pd.read_csv(
                f"{folder}{dataset}Missing_{int(params.miss_rate * 100)}.csv"
            )
            missing = df_missing.values

            mask = np.where(np.isnan(missing), 0.0, 1.0)
            missing_data = np.where(mask, missing, 0.0)
            hint = generate_hint(mask, params.hint_rate)
            # missing_data = missing_data[:2500]
            # mask = mask[:2500]
            range_scaler = (0, 1)
            scaler = MinMaxScaler(feature_range=range_scaler)
            missing_data = scaler.fit_transform(missing_data)
            data = scaler.transform(data)
            # mask = scaler.transform(mask)
            # hint = scaler.transform(hint)

            dim = missing_data.shape[1]
            size = missing_data.shape[0]

            missing_data = torch.from_numpy(missing_data)
            mask = torch.from_numpy(mask)
            hint = torch.from_numpy(hint)

            # combined_dataset = torch.utils.data.TensorDataset(missing_data, mask, hint)

            train_size = int(size * params.train_ratio)
            train_data, test_data = torch.utils.data.random_split(
                combined_dataset, [train_size, size - train_size]
            )

            h_dim1 = dim
            h_dim2 = dim

            for run in range(params.num_runs):
                start_run_time = time.time()
                print(
                    "\n\nStarting run number", run + 1, "out of", params.num_runs, "\n"
                )

                data_iter = torch.utils.data.DataLoader(
                    train_data, params.batch_size, pin_memory=True, num_workers=8
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
                    data,
                    train_data,
                    test_data,
                    missing_data,
                    params.alpha,
                    mask,
                    run,
                )

                run_time[run] = time.time() - start_run_time

        utils.create_csv(
            run_time,
            folder
            + "results/"
            + f"run_time_{int(params.miss_rate * 100)}_{train_size}",
            "Time for each run in seconds",
        )

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
