import Globals

from datetime import datetime, date

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper


class RewardWrapper(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

        if rewards[0] >= 1:
            Globals.blocks += 1

        Globals.results_list.append(
            {
                'timestamp': '{0}'.format(datetime.now()),
                'blocks': Globals.blocks,
                'number of rewards': Globals.given_rewards
            }
        )

        rewards[0] = Globals.reward
        Globals.reward = 0.0
        return observations, rewards, dones, infos