import torch

from RL_IRS_env import IRSEnv
from MLP import MultiLayerPerceptron as MLP
from DQN import DQN, prepare_training_inputs
from memory.memory import ReplayMemory
from train_utils import to_tensor

# Simulation parameter

phase_bit = 1  # Num of available phase
phase_N = 2**phase_bit
IRS_w = 2  # IRS element 2 x 2
element_N = IRS_w**2


# Hyperparameter
lr = 1e-4 * 5
batch_size = 3
gamma = 1.0
memory_size = 50000
total_eps = 3000
eps_max = 0.08
eps_min = 0.01
sampling_only_until = 10
target_update_interval = 5

qnet = MLP(element_N, phase_N ^ element_N, num_neurons=[256])
qnet_target = MLP(element_N, phase_N ^ element_N, num_neurons=[256])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(element_N, element_N, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)  # action_dim?
env = IRSEnv(IRS_w, phase_bit)
memory = ReplayMemory(memory_size)
print_every = 100

for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = max(eps_min, eps_max - eps_min * (n_epi / 200))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()
    cum_r = 0
    i = 0
    while True:

        s = to_tensor(s, size=(1, element_N))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r / 100.0).view(1, 1),
                      torch.tensor(ns).view(1, element_N),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)

        s = ns
        cum_r += r
        if done:
            break


    if len(memory) >= sampling_only_until:
        # train agent
        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

    if n_epi % target_update_interval == 0:
        qnet_target.load_state_dict(qnet.state_dict())

    if n_epi % print_every == 0:
        msg = (n_epi, r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))
