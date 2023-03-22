import numpy as np
from math import pi
from itertools import product


def quantiz(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta


def Q_BCD(Mc, Q=0):
    N_IRS = len(Mc) - 1

    # Original BCD
    idx = range(0, 5)
    wt = np.ones((N_IRS + 1, len(idx) + 1), complex)
    for itr in idx:
        wt0 = Mc @ (wt[:, itr].reshape(-1, 1))
        # PAPC
        wt[:, itr + 1] = (wt0 / np.abs(wt0)).ravel()

    psi_bcd = wt[:, -1]
    psi_bcd = psi_bcd / psi_bcd[-1]

    chi_bcd = np.angle(psi_bcd)

    if Q == 0:
        psi = psi_bcd[0:N_IRS]

    else:
        phase_set = np.linspace(-np.pi, np.pi, 2 ** Q + 1)
        threshold = (phase_set[0:-1] + phase_set[1:]) / 2
        _, q_chi = quantiz(chi_bcd[0:N_IRS], threshold, phase_set)
        q_chi = np.array(q_chi)
        q_psi = np.exp(1j * q_chi.T)
        psi = q_psi
    psi_1 = np.append(psi.conjugate().T, 1).reshape(1, -1)
    gamma = np.abs(psi_1 @ Mc @ (psi_1.conjugate().T))
    return psi, gamma


def make_action(b=2, N=4):
    phase_list = np.linspace(-pi, pi, 2**b, endpoint=False)
    action_list = list(product(phase_list, repeat=N))
    return action_list
