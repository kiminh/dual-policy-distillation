#!/usr/bin/env python3
import os
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

from multiprocessing import Pool
import csv
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def train(env_id, num_timesteps, seed, exp_scale):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.dpd_ppo import ppo2
    from baselines.dpd_ppo.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make(seed):
        def make_env():
            env = gym.make(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        env = DummyVecEnv([make_env])
        env = VecNormalize(env)
        return env

    set_global_seeds(seed)
    policy = MlpPolicy
    env_0 = make(seed)
    env_1 = make(seed+10000)
    
    #learner = ppo2.Learner(policy=policy, env=env_0, nsteps=2048, nminibatches=32,
    #                   lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
    #                   ent_coef=0.0,
    #                   lr=3e-4,
    #                   cliprange=0.2,
    #                   total_timesteps=num_timesteps/2,
    #                   scope='agent_0',
    #                   policy_scope='',
    #                   value_scope='')

    learner_0 = ppo2.Learner(policy=policy, env=env_0, nsteps=2048, nminibatches=64,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=5e-5,
                       cliprange=0.2,
                       total_timesteps=num_timesteps/2,
                       scope='agent_0',
                       policy_scope='0',
                       value_scope='0',
                       im_batch=512,
                       exp_scale=exp_scale,
                       )

    learner_1 = ppo2.Learner(policy=policy, env=env_1, nsteps=2048, nminibatches=64,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=5e-5,
                       cliprange=0.2,
                       total_timesteps=num_timesteps/2,
                       scope='agent_1',
                       policy_scope='1',
                       value_scope='1',
                       im_batch=512,
                       exp_scale=exp_scale
                       )
    train_flag = True
    reward = None
    while train_flag:
        if not learner_0.update():
            train_flag = False
        elif learner_0.update_index % 5 == 0:
            for _ in range(1):
                obs, actions, values = learner_1.sample(512)
                #learner_0.imitate(obs, actions, values)
                learner_0.imitate_mse(obs, actions, values)
                
        if not learner_1.update():
            train_flag = False
        elif learner_1.update_index % 5 == 0:
            for _ in range(1):
                obs, actions, values = learner_0.sample(512)
                #learner_1.imitate(obs, actions, values)
                learner_1.imitate_mse(obs, actions, values)
        #if learner_0.update_index % 20 == 0:
        #    reward_0 = learner_0.min_reward
        #    reward_1 = learner_1.min_reward
        #    max_reward_0 = learner_0.max_reward
        #    max_reward_1 = learner_1.max_reward
        #    if reward == None:
        #        rewards = [reward_0, reward_1]
        #    else:
        #        rewards = [reward_0, reward_1]
        #    best = rewards.index(max(rewards))
        #    print('Reward_0: ', reward_0, max_reward_0, '\tReward_1: ', reward_1, max_reward_1, '\tReward: ', reward, '\tBest:', best)
        #    if best == 0:
        #        learner_1.copy('model0')
        #        learner.copy('model0')
        #        reward = reward_0
        #    elif best == 1:
        #        learner_0.copy('model1')
        #        learner.copy('model1')
        #        reward = reward_1
        #    else:
        #        learner_0.copy('model')
        #        learner_1.copy('model')
        #    print('New reward: ', reward)
        ##learner_0.model.copy('model1')

def run(pair):
    env_id = pair[0]
    seed = pair[1]
    log_root = pair[2]
    num_timesteps = pair[3]
    exp_scale = pair[4]
    log_path = os.path.join(log_root, env_id+'_'+str(seed))
    logger.configure(dir=log_path)
    train(env_id, num_timesteps, seed, exp_scale)

def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--log-dir', type=str, default='./log', help="Log directory")
    parser.add_argument('--exp-scale', type=float, default=0.5, help="Exp scale of confidence score")
    args = parser.parse_args()
    #fig_path = os.path.join(args.log_dir, args.env_id+'.png')

    # Single processing for testing
    #arguments = [args.env, 0, args.log_dir, args.num_timesteps, args.exp_scale]
    #run(arguments)
    exp_num = 3
    # Multiprocessing
    pool = Pool(processes=exp_num)
    arguments = [[args.env_id, seed, args.log_dir, args.num_timesteps, args.exp_scale] for seed in range(exp_num)]
    pool.map(run, arguments)
    #stats_dict = {'timestep': [], 'reward': []}
    stats_dict = {'timestep': [], 'reward': [], 'agent': []}

    # Read Logs
    print('Reading logs...')
    for seed in range(exp_num):
        filename = os.path.join(args.log_dir, args.env_id+'_'+str(seed), 'progress.csv')
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)
            for row in csvreader:
                reward = row[fields.index('eprewmean')]
                timestep = row[fields.index('total_timesteps')]
                agent = row[fields.index('agent')]
                stats_dict['timestep'].append(int(timestep)*2)
                stats_dict['reward'].append(float(reward))
                stats_dict['agent'].append(agent)
    #print('Plotting results...')
    #data = pd.DataFrame(data=stats_dict)
    ##sns_plot = sns.lineplot(x="timestep", y="reward", hue='agent', data=data)
    ##sns_plot = sns.lineplot(x="timestep", y="reward", data=data)
    ##sns_plot.figure.savefig(fig_path)
    #plt.close(sns_plot.figure)
    

if __name__ == '__main__':
    main()
