import os
import time
from collections import deque
import pickle

from baselines.dpd_ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


class Learner(object):

    def __init__(self, sess, prefix, env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
        normalize_returns, normalize_observations, critic_l2_reg, actor_lr, actor_dis_lr, critic_lr, exp_scale, action_noise,
        popart, gamma, clip_norm, nb_train_steps, nb_dis_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, dis_batch_size, memory,
        tau=0.01, eval_env=None, param_noise_adaption_interval=50):

        self.sess = sess
        self.prefix = prefix
        self.env = env
        self.nb_epochs = nb_epochs
        self.nb_epoch_cycles = nb_epoch_cycles
        self.render_eval = render_eval
        self.reward_scale = reward_scale
        self.render = render
        self.param_noise = param_noise
        self.actor = actor
        self.critic = critic
        self.normalize_returns = normalize_returns
        self.normalize_observations = normalize_observations
        self.critic_l2_reg = critic_l2_reg
        self.actor_lr = actor_lr
        self.actor_dis_lr = actor_dis_lr
        self.critic_lr = critic_lr
        self.exp_scale = exp_scale
        self.action_noise = action_noise
        self.popart = popart
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.nb_train_steps = nb_train_steps
        self.nb_dis_train_steps = nb_dis_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.nb_eval_steps = nb_eval_steps
        self.batch_size = batch_size
        self.dis_batch_size = dis_batch_size
        self.memory = memory
        self.tau = tau
        self.eval_env = eval_env
        self.param_noise_adaption_interval = param_noise_adaption_interval
    
        self.rank = MPI.COMM_WORLD.Get_rank()

        assert (np.abs(self.env.action_space.low) == self.env.action_space.high).all()  # we assume symmetric actions.
        self.max_action = self.env.action_space.high
        logger.info('scaling actions by {} before executing in env'.format(self.max_action))

        self.agent = DDPG(self.prefix, self.actor, self.critic, self.memory, self.env.observation_space.shape, self.env.action_space.shape,
        gamma=self.gamma, tau=self.tau, normalize_returns=self.normalize_returns, normalize_observations=self.normalize_observations,
        batch_size=self.batch_size, dis_batch_size= self.dis_batch_size, action_noise=self.action_noise, param_noise=self.param_noise, critic_l2_reg=self.critic_l2_reg,
        actor_lr=self.actor_lr, actor_dis_lr=self.actor_dis_lr, critic_lr=self.critic_lr, exp_scale=self.exp_scale, enable_popart=self.popart, clip_norm=self.clip_norm,
        reward_scale=self.reward_scale)
        logger.info('Using agent with the following configuration:')
        logger.info(str(self.agent.__dict__.items()))

        # Set up logging stuff only for a single worker.
        if self.rank == 0:
            self.saver = tf.train.Saver()
        else:
            self.saver = None

        self.step = 0
        self.episode = 0
        self.eval_episode_rewards_history = deque(maxlen=100)
        self.episode_rewards_history = deque(maxlen=100)
        #with U.single_threaded_session() as sess:
        #self.sess = U.single_threaded_session()
        # Prepare everything.
        self.agent.initialize(self.sess)
        #self.sess.graph.finalize()


        self.agent.reset()
        self.obs = self.env.reset()
        if self.eval_env is not None:
            self.eval_obs = eval_env.reset()
        self.done = False
        self.episode_reward = 0.
        self.episode_step = 0
        self.episodes = 0
        self.t = 0

        self.epoch = 0
        self.start_time = time.time()

        self.epoch_episode_rewards = []
        self.epoch_episode_steps = []
        self.epoch_episode_eval_rewards = []
        self.epoch_episode_eval_steps = []
        self.epoch_start_time = time.time()
        self.epoch_actions = []
        self.epoch_qs = []
        self.epoch_episodes = 0

        self.epoch_actor_losses = []
        self.epoch_critic_losses = []
        self.epoch_adaptive_distances = []
        self.eval_episode_rewards = []
        self.eval_qs = []


    def rollout(self):
        # Perform rollouts.
        for t_rollout in range(self.nb_rollout_steps):
            # Predict next action.
            action, q = self.agent.pi(self.obs, apply_noise=True, compute_Q=True)
            assert action.shape == self.env.action_space.shape

            # Execute next action.
            if self.rank == 0 and self.render:
                self.env.render()
            assert self.max_action.shape == action.shape
            new_obs, r, self.done, info = self.env.step(self.max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            self.t += 1
            if self.rank == 0 and self.render:
                self.env.render()
            self.episode_reward += r
            self.episode_step += 1

            # Book-keeping.
            self.epoch_actions.append(action)
            self.epoch_qs.append(q)
            self.agent.store_transition(self.obs, action, r, new_obs, self.done)
            self.obs = new_obs

            if self.done:
                # Episode done.
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_rewards_history.append(self.episode_reward)
                self.epoch_episode_steps.append(self.episode_step)
                self.episode_reward = 0.
                self.episode_step = 0
                self.epoch_episodes += 1
                self.episodes += 1

                self.agent.reset()
                self.obs = self.env.reset()

    def train(self):
        # Train.
        self.epoch_actor_losses = []
        self.epoch_critic_losses = []
        self.epoch_adaptive_distances = []
        self.epoch_dis_losses = []
        for t_train in range(self.nb_train_steps):
            # Adapt param noise, if necessary.
            if self.memory.nb_entries >= self.batch_size and t_train % self.param_noise_adaption_interval == 0:
                distance = self.agent.adapt_param_noise()
                self.epoch_adaptive_distances.append(distance)

            cl, al = self.agent.train()
            dis_loss = self.agent.dis_train()
            self.epoch_critic_losses.append(cl)
            self.epoch_actor_losses.append(al)
            self.epoch_dis_losses.append(dis_loss)

            self.agent.update_target_net()
        
    def evaluate(self):
        # Evaluate.
        self.eval_episode_rewards = []
        self.eval_qs = []
        if self.eval_env is not None:
            eval_episode_reward = 0.
            for t_rollout in range(self.nb_eval_steps):
                eval_action, eval_q = self.agent.pi(self.eval_obs, apply_noise=False, compute_Q=True)
                self.eval_obs, eval_r, self.eval_done, eval_info = self.eval_env.step(self.max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                if self.render_eval:
                    self.eval_env.render()
                eval_episode_reward += eval_r

                self.eval_qs.append(eval_q)
                if self.eval_done:
                    self.eval_obs = self.eval_env.reset()
                    self.eval_episode_rewards.append(eval_episode_reward)
                    self.eval_episode_rewards_history.append(eval_episode_reward)
                    eval_episode_reward = 0.

    #def dis_train(self):
    #    self.epoch_dis_losses = []
    #    for t_train in range(self.nb_dis_train_steps):
    #        print(self.nb_dis_train_steps)
    #        dis_loss = self.agent.dis_train()
    #        self.epoch_dis_losses.append(dis_loss)
        

    def update(self):
        #for epoch in range(self.nb_epochs):
        for cycle in range(self.nb_epoch_cycles):
            self.rollout()
            self.train()
            # Distillation
            #self.dis_train()
            # Distillation
            self.evaluate()


        # Logging
        self.do_log()
        self.epoch += 1
        

    def do_log(self):
        mpi_size = MPI.COMM_WORLD.Get_size()
        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - self.start_time
        stats = self.agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(self.epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(self.episode_rewards_history)
        combined_stats['rollout/return_lateset'] = self.episode_rewards_history[-1]
        combined_stats['rollout/episode_steps'] = np.mean(self.epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(self.epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(self.epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(self.epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(self.epoch_critic_losses)
        combined_stats['train/loss_distillation'] = np.mean(self.epoch_dis_losses)
        combined_stats['train/param_noise_distance'] = np.mean(self.epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(self.t) / float(duration)
        combined_stats['total/episodes'] = self.episodes
        combined_stats['rollout/episodes'] = self.epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(self.epoch_actions)
        # Evaluation statistics.
        if self.eval_env is not None:
            combined_stats['eval/return'] = np.mean(self.eval_episode_rewards)
            combined_stats['eval/return_history'] = np.mean(self.eval_episode_rewards_history)
            combined_stats['eval/episode_return'] = self.eval_episode_rewards_history[-1]
            combined_stats['eval/Q'] = np.mean(self.eval_qs)
            combined_stats['eval/episodes'] = len(self.eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)
        #print(combined_stats.values())
        combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = self.epoch + 1
        combined_stats['total/steps'] = self.t
        combined_stats['learner'] = self.prefix

        for key in sorted(combined_stats.keys()):
            #logger.record_tabular(key, combined_stats[key])
            logger.logkv(key, combined_stats[key])
        #logger.dump_tabular()
        logger.dumpkvs()
        logger.log('')
        logdir = logger.get_dir()
        if self.rank == 0 and logdir:
            if hasattr(self.env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(self.env.get_state(), f)
            if self.eval_env and hasattr(self.eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(self.eval_env.get_state(), f)

    def get_agent(self):
        return self.agent

    def set_partner_agent(self, agent):
        self.agent.set_partner_agent(agent)
