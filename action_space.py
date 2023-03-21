import numpy as np
from math import pi
from itertools import product


def make_action(b=2, N=4):
    phase_list = np.linspace(0, 2*pi, 2^b, endpoint=False)
    action_list = list(product(phase_list, repeat=N))
    print(phase_list)
    print(action_list)
    return action_list


make_action()







