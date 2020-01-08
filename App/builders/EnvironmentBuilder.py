import Globals

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from custom.RewardWrapper import RewardWrapper

id = 'BreakoutNoFrameskip-v4'
num_env = 1
seed = 0
n_stack = 4


def init():
    env = make_atari_env(id, num_env=num_env, seed=seed)
    env = VecFrameStack(env, n_stack=n_stack)
    Globals.env = RewardWrapper(env)