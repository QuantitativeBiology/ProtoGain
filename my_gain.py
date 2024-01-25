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




def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0., 1., size=(size,dim))
    B = A > miss_rate
    mask = 1. * B

    return mask

def generate_hint(mask, hint_rate):
    hint_mask = generate_mask(mask, 1-hint_rate)
    hint = mask * hint_mask 

    return hint


def update_D(batch, mask, hint, Z, net_D, net_G, loss, optimizer_D):
    new_X = mask * batch + (1-mask) * Z
    
    input_G = torch.cat((new_X, mask), 1).float()

    sample_G = net_G(input_G)
    fake_X = new_X * mask + sample_G * (1-mask)
    fake_input_D = torch.cat((fake_X.detach(), hint), 1).float()
    fake_Y = net_D(fake_input_D)
    
    loss_D = (loss(fake_Y.float(), mask.float()) ).mean()

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    return loss_D

def update_G(batch, mask, hint, Z, net_D, net_G, loss, optimizer_G, alpha):
    
    loss_mse = nn.MSELoss(reduction = 'none')

    ones = torch.ones_like(batch)

    new_X = mask * batch + (1-mask) * Z 
    input_G = torch.cat((new_X, mask), 1).float()
    sample_G = net_G(input_G)
    fake_X = new_X * mask + sample_G * (1-mask)

    fake_input_D = torch.cat((fake_X, hint), 1).float()
    fake_Y = net_D(fake_input_D)

    #print(batch, mask, ones.reshape(fake_Y.shape), fake_Y, loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask), (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask)).mean())
    loss_G_entropy = (loss(fake_Y, ones.reshape(fake_Y.shape).float()) * (1-mask) ).mean() 
    loss_G_mse = (loss_mse((sample_G*mask).float(), (batch*mask).float())).mean() 
    
    loss_G = loss_G_entropy + alpha * loss_G_mse

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    return loss_G


def train(net_D, net_G, lr_D, lr_G, data_iter, num_epochs, 
          data, missing_data, alpha, mask, run):
    
    dim = missing_data.shape[1]
    size = missing_data.shape[0]
    
    #loss = nn.BCEWithLogitsLoss(reduction = 'sum')
    loss = nn.BCELoss(reduction = 'none')
    loss_mse = nn.MSELoss(reduction = 'none')

    loss_D_values = np.zeros(num_epochs)
    loss_G_values = np.zeros(num_epochs)
    loss_MSE_values = np.zeros(num_epochs)
     
    #for w in net_D.parameters():
    #    nn.init.normal_(w, 0, 0.02)
    #for w in net_G.parameters():
    #    nn.init.normal_(w, 0, 0.02)

    #for w in net_D.parameters():
    #    nn.init.xavier_normal_(w)
    #for w in net_G.parameters():
    #    nn.init.xavier_normal_(w)

    # Initialize weights for net_D
    for name, param in net_D.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
            #nn.init.uniform_(param)

    # Initialize weights for net_G
    for name, param in net_G.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
            #nn.init.uniform_(param)

    #optimizer_D = torch.optim.SGD(net_D.parameters(), lr = lr_D)
    #optimizer_G = torch.optim.SGD(net_G.parameters(), lr = lr_G)

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr = lr_D)
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr = lr_G)

    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        
        for batch, mask_batch, hint_batch in data_iter:
            batch_size = batch.shape[0]

            #Z = torch.normal(0, 1, size=(batch_size, dim))
            Z = torch.rand((batch_size, dim)) * 0.01
            loss_D = update_D(batch, mask_batch, hint_batch, Z, net_D, net_G, 
                              loss, optimizer_D)
            loss_G = update_G(batch, mask_batch, hint_batch, Z, net_D, net_G, 
                              loss, optimizer_G, alpha)

        if epoch % 100 == 0:
            s = "{:6d}) loss D {:0.3f} loss G {:0.3f}".format(
                epoch,
                loss_D.detach().numpy(),
                loss_G.detach().numpy())
            pbar.clear()
            #logger.info('{}'.format(s))
            pbar.set_description(s)
            
        Z = torch.rand((size, dim)) * 0.01
        missing_data_with_noise = mask * missing_data + (1-mask) * Z 
        input_G = torch.cat((missing_data_with_noise, mask), 1).float()
        sample_G = net_G(input_G)

        loss_D_values[epoch] = loss_D.detach().numpy()
        loss_G_values[epoch] = loss_G.detach().numpy()
        loss_MSE_values[epoch] = ( loss_mse((1 - mask) * data, (1 - mask) *  sample_G) ).mean() / (1-mask).mean()

        #print("Data:\n", data, scaler.inverse_transform(data), "\nImputed:\n", fake_X, scaler.inverse_transform(fake_X.detach().numpy()))

    data_imputed = missing_data * mask + sample_G * (1-mask)
    data_imputed = scaler.inverse_transform(data_imputed.detach().numpy())

    utils.create_csv(data_imputed, folder + "results/" + dataset + "Imputed_{}_{}".format(int(params.miss_rate * 100), run), data_header)
    utils.create_csv(loss_D_values, folder + "results/lossD_{}_{}".format(int(params.miss_rate * 100), run), "loss D")
    utils.create_csv(loss_G_values, folder + "results/lossG_{}_{}".format(int(params.miss_rate * 100), run), "loss G")
    utils.create_csv(loss_MSE_values, folder + "results/lossMSE_{}_{}".format(int(params.miss_rate * 100), run), "loss MSE")

    return loss_MSE_values[epoch]





if __name__ == "__main__":
    
    with cProfile.Profile() as profile:
        dataset = "breast"
        folder = "~/LeandroSobralThesis/" + dataset + "/"

        params = Params.read_hyperparameters("parameters.json")
        loss_MSE_final = np.zeros(params.num_runs)

        df_data = pd.read_csv(folder + dataset + ".csv")
        #features = list(df_data.columns)
        data = df_data.values
        data_header = df_data.columns.tolist()


        df_missing = pd.read_csv(folder + dataset 
                                 + "Missing_{}".format(int(params.miss_rate * 100)) + ".csv")
        missing = df_missing.values

        mask = np.where(np.isnan(missing), 0.0, 1.0)
        missing_data = np.where(mask, missing, 0.0)
        hint = generate_hint(mask, params.hint_rate)
        #missing_data = missing_data[:2500]
        #mask = mask[:2500]

        range_scaler = (0, 1)
        scaler = MinMaxScaler(feature_range=range_scaler)
        missing_data = scaler.fit_transform(missing_data)
        data = scaler.transform(data)
        #mask = scaler.transform(mask)
        #hint = scaler.transform(hint)

        dim = missing_data.shape[1]
        size = missing_data.shape[0]

        missing_data = torch.from_numpy(missing_data)
        mask = torch.from_numpy(mask)
        hint = torch.from_numpy(hint)

        combined_dataset = torch.utils.data.TensorDataset(missing_data, mask, hint)

        h_dim1 = dim
        h_dim2 = dim

        for run in range(params.num_runs):

            print("\n\nStarting run number", run+1, "out of", params.num_runs, "\n")

            data_iter = torch.utils.data.DataLoader(combined_dataset, params.batch_size, shuffle=True)

            net_G = nn.Sequential(
                nn.Linear(dim*2, h_dim1), nn.ReLU(),
                nn.Linear(h_dim1, h_dim2), nn.ReLU(),
                nn.Linear(h_dim2, dim), nn.Sigmoid())

            net_D = nn.Sequential(
                nn.Linear(dim*2, h_dim1), nn.ReLU(),
                nn.Linear(h_dim1, h_dim2), nn.ReLU(),
                nn.Linear(h_dim2, dim), nn.Sigmoid())

            loss_MSE_final[run] = train(net_D, net_G, params.lr_D, params.lr_G, data_iter, 
                params.num_epochs, data, missing_data, params.alpha, mask, run)
            
            print("Final MSE =", loss_MSE_final[run])
            

        utils.create_csv(loss_MSE_final, folder + "results/lossMSEfinal_{}".format(int(params.miss_rate * 100)), "Final MSE")
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    #results.print_stats()
    results.dump_stats("results.prof")