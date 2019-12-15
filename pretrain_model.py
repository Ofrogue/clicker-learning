import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import gym

from stable_baselines.common.cmd_util import make_atari_env # rl-zoo model is custom in contrast to gym defaults

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

from stable_baselines.common.vec_env import VecFrameStack

import pickle

from myCnnPolicy import MyCnnPolicy
from myDQN import MyDQN

env_id = 'BreakoutNoFrameskip-v4'
env = make_atari_env(env_id, num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)

def load_zoo_model():
    model = DQN(CnnPolicy, env, double_q=False, learning_starts=10, tensorboard_log='./tensor_files/',verbose=2)
    file = open('BreakoutNoFrameskip-v4.pkl', 'rb')
    model_dict, model_weights = pickle.load(file)
    model.load_parameters(model_weights)
    return model

def get_zoo_parameters():
    """loads all available params from BreakoutNoFrameskip-v4.pkl 
    privided by https://github.com/araffin/rl-baselines-zoo
    """
    model = load_zoo_model()
    return model.get_parameters()

def load_custom_model(*args, **kwargs):
    model = MyDQN(MyCnnPolicy, env, *args, **kwargs)
    model.load_parameters(get_zoo_parameters(), exact_match=False)
    return model

model = load_custom_model(double_q=False, 
                          learning_starts=1000, 
                          tensorboard_log='./tensor_files/', 
                          verbose=2,
                          prioritized_replay=True
                         )
model.learn(200000)
model.save('./pretrained_model.zip')




