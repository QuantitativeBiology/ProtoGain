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
    parser.add_argument("-i", help="path to missing data")
    parser.add_argument("-o", default="imputed", help="name of output file")
    parser.add_argument("--ref", help="path to a reference (complete) dataset")
    parser.add_argument(
        "--ofolder", default=os.getcwd() + "/results/", help="path to output folder"
    )
    parser.add_argument("--it", type=int, default=2000, help="number of iterations")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--alpha", type=float, default=10, help="alpha")
    parser.add_argument("--hint", type=float, default=0.8, help="hint rate")
    parser.add_argument(
        "--trainratio", help="percentage of data to be used as a train set"
    )
    parser.add_argument(
        "--lrd", type=float, default=0.001, help="learning rate for the discriminator"
    )
    parser.add_argument(
        "--lrg", type=float, default=0.001, help="learning rate for the generator"
    )
    parser.add_argument("--parameters", help="load a parameters.json file")
    parser.add_argument(
        "--override", type=int, default=0, help="override previous files"
    )
    return parser.parse_args()


def output(
    data_train_imputed,
    output_folder,
    output_file,
    missing_header,
    loss_D_values,
    loss_G_values,
    loss_MSE_train,
    loss_MSE_test,
    cpu,
    ram,
    ram_percentage,
    override,
):

    utils.create_csv(
        data_train_imputed,
        output_folder + output_file,
        missing_header,
    )
    utils.create_output(
        loss_D_values,
        output_folder + "lossD.csv",
        override,
    )
    utils.create_output(
        loss_G_values,
        output_folder + "lossG.csv",
        override,
    )
    utils.create_output(
        loss_MSE_train,
        output_folder + "lossMSE_train.csv",
        override,
    )

    utils.create_output(
        loss_MSE_test,
        output_folder + "lossMSE_test.csv",
        override,
    )

    utils.create_output(
        cpu,
        output_folder + "cpu.csv",
        override,
    )

    utils.create_output(ram, output_folder + "ram.csv", override)

    utils.create_output(
        ram_percentage,
        output_folder + "ram_percentage.csv",
        override,
    )


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


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def train_v2(
    net_D,
    net_G,
    lr_D,
    lr_G,
    alpha,
    num_iterations,
    batch_size,
    train_size,
    missing,
    mask,
    combined_dataset_train,
    output_file,
    missing_header,
    override,
):
    cpu = []
    ram = []
    ram_percentage = []

    dim = missing.shape[1]

    # loss = nn.BCEWithLogitsLoss(reduction = 'sum')
    loss = nn.BCELoss(reduction="none")
    loss_mse = nn.MSELoss(reduction="none")

    loss_D_values = np.zeros(num_iterations)
    loss_G_values = np.zeros(num_iterations)
    loss_MSE_train = np.zeros(num_iterations)
    loss_MSE_test = np.zeros(num_iterations)

    # for w in net_D.parameters():
    #    nn.init.normal_(w, 0, 0.02)
    # for w in net_G.parameters():
    #    nn.init.normal_(w, 0, 0.02)

    # for w in net_D.parameters():
    #    nn.init.xavier_normal_(w)
    # for w in net_G.parameters():
    #    nn.init.xavier_normal_(w)

    for name, param in net_D.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    for name, param in net_G.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
            # nn.init.uniform_(param)

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)

    pbar = tqdm(range(num_iterations))
    for it in pbar:

        mb_idx = sample_idx(train_size, batch_size)

        batch = torch.stack([combined_dataset_train[idx][0] for idx in mb_idx])
        mask_batch = torch.stack([combined_dataset_train[idx][1] for idx in mb_idx])
        hint_batch = torch.stack([combined_dataset_train[idx][2] for idx in mb_idx])
        ref_batch = torch.stack([combined_dataset_train[idx][3] for idx in mb_idx])

        Z = torch.rand((batch_size, dim)) * 0.01
        loss_D = update_D(
            batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_D
        )
        loss_G = update_G(
            batch, mask_batch, hint_batch, Z, net_D, net_G, loss, optimizer_G, alpha
        )

        sample_G = generate_sample(batch, mask_batch)

        loss_MSE_train[it] = (
            loss_mse(mask_batch * batch, mask_batch * sample_G)
        ).mean()

        loss_MSE_test[it] = (
            loss_mse((1 - mask_batch) * ref_batch, (1 - mask_batch) * sample_G)
        ).mean() / (1 - mask_batch).mean()

        if it % 100 == 0:
            s = f"{it}: loss D={loss_D.detach().numpy(): .3f}  loss G={loss_G.detach().numpy(): .3f}  rmse train={np.sqrt(loss_MSE_train[it]): .4f}  rmse test={np.sqrt(loss_MSE_test[it]): .3f}"
            pbar.clear()
            pbar.set_description(s)

        cpu.append(psutil.cpu_percent())
        ram.append(psutil.virtual_memory()[3] / 1000000000)
        ram_percentage.append(psutil.virtual_memory()[2])

        loss_D_values[it] = loss_D.detach().numpy()
        loss_G_values[it] = loss_G.detach().numpy()

    sample_G = generate_sample(missing, mask)
    data_train_imputed = missing * mask + sample_G * (1 - mask)
    data_train_imputed = scaler.inverse_transform(data_train_imputed.detach().numpy())

    output(
        data_train_imputed,
        output_folder,
        output_file,
        missing_header,
        loss_D_values,
        loss_G_values,
        loss_MSE_train,
        loss_MSE_test,
        cpu,
        ram,
        ram_percentage,
        override,
    )


if __name__ == "__main__":
    start_time = time.time()
    with cProfile.Profile() as profile:

        folder = os.getcwd()

        args = init_arg()

        missing_file = args.i
        output_file = args.o
        ref_file = args.ref
        output_folder = args.ofolder
        num_iterations = args.it
        batch_size = args.batchsize
        alpha = args.alpha
        hint_rate = args.hint
        train_ratio = args.trainratio
        lr_D = args.lrd
        lr_G = args.lrg
        parameters_file = args.parameters
        override = args.override

        if parameters_file is not None:
            params = Params.read_hyperparameters(parameters_file)
            missing_file = params.input
            output_file = params.output
            ref_file = params.ref
            output_folder = params.output_folder
            num_iterations = params.num_iterations
            batch_size = params.batch_size
            alpha = params.alpha
            hint_rate = params.hint_rate
            train_ration = params.train_ratio
            lr_D = params.lr_D
            lr_G = params.lr_G
            override = params.override

        df_missing = pd.read_csv(missing_file)
        missing = df_missing.values
        missing_header = df_missing.columns.tolist()

        df_ref = pd.read_csv(ref_file)
        ref = df_ref.values
        ref_header = df_ref.columns.tolist()

        dim = missing.shape[1]
        train_size = missing.shape[0]

        if dim != ref.shape[1]:
            print(
                "\n\nThe reference and data files provided don't have the same number of features\n"
            )
            exit(1)
        elif train_size != ref.shape[0]:
            print(
                "\n\nThe reference and data files provided don't have the same number of samples\n"
            )
            exit(2)

        mask = np.where(np.isnan(missing), 0.0, 1.0)
        missing = np.where(mask, missing, 0.0)
        hint = generate_hint(mask, hint_rate)

        range_scaler = (0, 1)
        scaler = MinMaxScaler(feature_range=range_scaler)
        missing = scaler.fit_transform(missing)
        ref = scaler.transform(ref)

        missing = torch.from_numpy(missing)
        mask = torch.from_numpy(mask)
        hint = torch.from_numpy(hint)
        ref = torch.from_numpy(ref)

        combined_dataset_train = torch.utils.data.TensorDataset(
            missing, mask, hint, ref
        )

        h_dim1 = dim
        h_dim2 = dim

        data_iter = torch.utils.data.DataLoader(
            missing, batch_size, pin_memory=True, num_workers=8
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
            lr_D,
            lr_G,
            alpha,
            num_iterations,
            batch_size,
            train_size,
            data_iter,
            ref,
            missing,
            mask,
            combined_dataset_train,
            output_file,
            missing_header,
            override,
        )

        run_time = []
        run_time.append(time.time() - start_time)
        file_path = output_folder + "run_time.csv"

        if override == 1:
            df_run_time = pd.DataFrame(run_time)
            df_run_time.to_csv(file_path, index=False)

        else:
            if os.path.exists(file_path):
                with open(file_path, "a") as myfile:
                    myfile.write(str(run_time[0]) + "\n")

            else:
                df_run_time = pd.DataFrame(run_time)
                df_run_time.to_csv(file_path, index=False)

    print("\n--- %s seconds ---\n\n" % (run_time[0]))
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    results.dump_stats("results.prof")
