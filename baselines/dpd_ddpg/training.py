from baselines.dpd_ddpg.learner import Learner
from baselines import logger
import baselines.common.tf_util as U
from baselines.dpd_ddpg.models import Actor, Critic
from baselines.dpd_ddpg.memory import Memory


def train(env_0, env_1, layer_norm, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise,
            normalize_returns, normalize_observations, critic_l2_reg, actor_lr, actor_dis_lr, critic_lr, exp_scale, action_noise,
                popart, gamma, clip_norm, nb_train_steps, nb_dis_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, dis_batch_size,
                tau=0.01, eval_env=None, param_noise_adaption_interval=50):

    dis_memory_size = nb_rollout_steps * nb_epoch_cycles
    print('Distillation Memory Size: ', dis_memory_size)

    sess = U.single_threaded_session()
    sess.__enter__()

    nb_actions = env_0.action_space.shape[-1]

    actor_0 = Actor(nb_actions, name='actor_0', layer_norm=layer_norm)
    critic_0 = Critic(name='critic_0', layer_norm=layer_norm)
    memory_0 = Memory(limit=int(1e6), action_shape=env_0.action_space.shape, observation_shape=env_0.observation_space.shape)
    
    learner_0 = Learner(sess, "0", env_0, nb_epochs/2, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor_0, critic_0,
                        normalize_returns, normalize_observations, critic_l2_reg, actor_lr, actor_dis_lr, critic_lr, exp_scale, action_noise,
                        popart, gamma, clip_norm, nb_train_steps, nb_dis_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, dis_batch_size, memory_0,
                        tau=tau, eval_env=eval_env, param_noise_adaption_interval=param_noise_adaption_interval)
    
    
    actor_1 = Actor(nb_actions, name='actor_1', layer_norm=layer_norm)
    critic_1 = Critic(name='critic_1', layer_norm=layer_norm)
    memory_1 = Memory(limit=int(1e6), action_shape=env_0.action_space.shape, observation_shape=env_1.observation_space.shape)
    learner_1 = Learner(sess, "1", env_1, nb_epochs/2, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor_1, critic_1,
                        normalize_returns, normalize_observations, critic_l2_reg, actor_lr, actor_dis_lr, critic_lr, exp_scale, action_noise,
                        popart, gamma, clip_norm, nb_train_steps, nb_dis_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, dis_batch_size, memory_1,
                        tau=tau, eval_env=eval_env, param_noise_adaption_interval=param_noise_adaption_interval)
    
    agent_0 = learner_0.get_agent()
    agent_1 = learner_1.get_agent()

    learner_0.set_partner_agent(agent_1)
    learner_1.set_partner_agent(agent_0)
    sess.graph.finalize()


    for epoch in range(int(nb_epochs/2)):

        logger.log('##### Epoch ', epoch, ', timesteps: ', learner_0.t * 2, ' #####')
        logger.log('> Learner 0')
        learner_0.update()

        logger.log('> Learner 1')
        learner_1.update()
