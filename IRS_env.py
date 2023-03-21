import numpy as np
import torch
import random

from action_space import make_action

class IRS_env:
    def __init__(self, N_IRS):
        # INITIALIZE MISO ANTENNAS PARAMETERS
        self.Nt = 4
        self.Nr = 1

        # Coordinate setting ([x,y,z])
        self.C_BS = np.array([0, 0, 10])
        self.C_IRS = np.array([250, 250, 5])
        self.C_UE = np.array([300, 200, 0])

        # enviroment parameter
        p_tx_dB = 30
        K_dB = 10  # Rician K (decibel) factor
        fc = 2.5  # carrier frequency [GHz]

        self.K = 10 ** (K_dB / 10)
        self.P_tx = 10 ** (p_tx_dB / 10)

        lambda_c = 3e8 / (fc * 1e9)  # carrier wavelength
        d = lambda_c / 2  # antenna element separation
        W = 10e6  # bandwidth 10MHz
        ts = 1 / W