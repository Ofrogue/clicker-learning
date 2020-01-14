import pickle, Globals
import numpy as np

from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

from custom.MyDQN import MyDQN
from custom.MyCnnPolicy import MyCnnPolicy

double_q = False
prioritized_replay = True
learning_starts = 1
tensorboard_log = 'tensor/'
verbose = 2
exploration_fraction = 0.0002
exploration_final_eps = 0.00002

magnitude = 0.1
learning_rate = 0.000001


def init():
    file = open('models/BreakoutNoFrameskip-v4.pkl', 'rb')
    _, zoo_weights = pickle.load(file)

    model = MyDQN(MyCnnPolicy, Globals.env, double_q=double_q, learning_starts=learning_starts, learning_rate=learning_rate,
                  tensorboard_log=tensorboard_log, verbose=verbose, exploration_fraction=exploration_fraction,
                  prioritized_replay=prioritized_replay, exploration_final_eps=exploration_final_eps)

    zoo_model = DQN(CnnPolicy, Globals.env, double_q=double_q, learning_starts=learning_starts)
    zoo_model.load_parameters(zoo_weights)

    model.load_parameters(zoo_model.get_parameters(), exact_match=False)
    params = model.get_parameters()
    r = (np.random.rand(4, 4) - 0.5) * magnitude
    params['deepq/model/action_value/fully_connected_1/biases:0'] = np.zeros(4)
    params['deepq/model/action_value/fully_connected_1/weights:0'] = np.identity(4) + r
    model.load_parameters(params)
    Globals.model = model
