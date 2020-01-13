import time

import Globals


from stable_baselines import DQN

from gym.spaces import MultiDiscrete
from stable_baselines.deepq.build_graph import build_act, build_act_with_param_noise

from functools import partial

import tensorflow as tf
import numpy as np
import gym
import logging
import os

from stable_baselines import logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack


class MyDQN(DQN):

    def __init__(self, *args, **kwargs):

        super(MyDQN, self).__init__(*args, **kwargs)

    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model = my_build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                    double_q=self.double_q
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("deepq")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []
            Globals.env = self.env
            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))
            timesteps_last_log = 0
            avr_ep_len_per_log = None
            sleep = 0.05

            for _ in range(total_timesteps):

                while Globals.paus_game:
                    pass

                if Globals.exit_learning:
                    break

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                      self.num_timesteps)

                episode_rewards[-1] += rew
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)

                if can_sample:
                    sleep = 0.03

                # time.sleep(sleep)

                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if self.prioritized_replay:
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                if len(episode_rewards) % log_interval == 0:
                    avr_ep_len_per_log = (self.num_timesteps - timesteps_last_log) / log_interval
                    timesteps_last_log = self.num_timesteps

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.record_tabular("avr length of last logged ep", avr_ep_len_per_log)
                    logger.dump_tabular()

                self.num_timesteps += 1

        return self

    def evaluate(self, n_episodes=2):

        logging.basicConfig(level=logging.INFO)

        id = 'BreakoutNoFrameskip-v4'
        num_env = 1
        n_stack = 4
        left_lives = 5
        seed = 0
        episodes = 0
        score = 0
        frames = 0
        frames_per_episode = list()
        scores = [list() for i in range(n_episodes)]

        env = make_atari_env(id, num_env=num_env, seed=seed)
        env = VecFrameStack(env, n_stack=n_stack)
        obs = env.reset()


        while (n_episodes - episodes) > 0:
            frames += 1
            action, _states = self.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            score += rewards[0]
            if dones:
                logging.debug('You died')
                logging.debug(f'Score = {score}')
                scores[episodes].append(score)
                score = 0
                left_lives -= 1
            if not left_lives:
                logging.debug('Episode ended')
                logging.info(f'Scores per life: {scores[episodes]}')
                frames_per_episode.append(frames)
                frames = 0
                episodes += 1
                left_lives = 5

        s = list(map(sum, scores))
        avg_s = int(sum(s) / len(s))
        avg_f = int(sum(frames_per_episode) / len(frames_per_episode))

        logging.info(f'Played {n_episodes} episodes')
        logging.info(f'Scores per episode : {s}')
        logging.info(f'Average score per episode : {avg_s}')
        logging.info(f'Average number of frames per episode : {avg_f}')

        return avg_f, avg_s



def my_build_train(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                   gamma=1.0, double_q=True, scope="deepq", reuse=None,
                   param_noise=False, param_noise_filter_func=None, full_tensorboard_log=False):
    """
    Creates the train function:
    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :return: (tuple)
        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    with tf.variable_scope(scope, reuse=reuse):
        if param_noise:
            act_f, obs_phs = build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                                                        param_noise_filter_func=param_noise_filter_func)
        else:
            act_f, obs_phs = build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess)

        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        my_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=tf.get_variable_scope().name + "/model/action_value/fully_connected_1")
        # target q network evaluation

        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_q_values = None
        double_obs_ph = target_policy.obs_ph
        if double_q:
            with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                double_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
                double_q_values = double_policy.q_values
                double_obs_ph = double_policy.obs_ph

    with tf.variable_scope("loss", reuse=reuse):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
            q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions),
                                       axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.scalar("loss", weighted_error)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # compute optimization op (potentially with gradient clipping)
        # print(q_func_vars)
        # func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model/action_value")
        # print(tf.get_variable_scope().name)
        print('Trainable tensors:')
        for v in my_q_func_vars:
            print(v)
        gradients = optimizer.compute_gradients(weighted_error, var_list=my_q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            if tf_util.is_image(obs_phs[0]):
                tf.summary.image('observation', obs_phs[0])
            elif len(obs_phs[0].shape) == 1:
                tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, td_error],
        updates=[optimize_expr]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr])

    return act_f, train, update_target, step_model






