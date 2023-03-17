import torch
import gym

from MLP import MultiLayerPerceptron as MLP
from DQN import DQN, prepare_training_inputs
from memory.memory import ReplayMemory
from train_utils import to_tensor

# Simulation parameter
##
# Hyperparameter
lr = 1e-4 * 5
batch_size = 256
gamma = 1.0
memory_size = 50000
total_eps = 3000
eps_max = 0.08
eps_min = 0.01
sampling_only_until = 2000
target_update_interval = 10

qnet = MLP(4, 2, num_neurons=[128])
qnet_target = MLP(4, 2, num_neurons=[128])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(4, 1, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0) # 하나의 agent로 각각 element에 action을 줄 수 없을듯.
env = gym.make('CartPole-v1') # have to change!
memory = ReplayMemory(memory_size)
print_every = 100

for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = max(eps_min, eps_max - eps_min * (n_epi / 200))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()
    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 4))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r / 100.0).view(1, 1),
                      torch.tensor(ns).view(1, 4),
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
        msg = (n_epi, cum_r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))