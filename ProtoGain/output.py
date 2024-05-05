from hypers import Params
import numpy as np


class Metrics:
    def __init__(self, hypers: Params):
        self.loss_D = np.zeros(hypers.num_iterations)
        self.loss_G = np.zeros(hypers.num_iterations)
        self.loss_MSE_train = np.zeros(hypers.num_iterations)
        self.loss_MSE_test = np.zeros(hypers.num_iterations)
        self.cpu = np.zeros(hypers.num_iterations)
        self.ram = np.zeros(hypers.num_iterations)
        self.ram_percentage = np.zeros(hypers.num_iterations)
        self.data_imputed = None
