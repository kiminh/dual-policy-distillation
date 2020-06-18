import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.runners import AbstractEnvRunner
from baselines.dpd_ppo.memory import Memory

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, policy_scope='', value_scope='', scope='', exp_scale=0.5, im_batch=512):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1,  policy_scope=policy_scope, value_scope=value_scope)
        im_model = policy(sess, ob_space, ac_space, im_batch, 1,  policy_scope=policy_scope, value_scope=value_scope)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, policy_scope=policy_scope, value_scope=value_scope)
        im_A = im_model.pdtype.sample_placeholder([None]) # A placeholder for imitation
        im_V = tf.placeholder(tf.float32, [None])
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        im_neglogpac = im_model.pd.neglogp(im_A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5, name='Adam'+scope)
        _train = trainer.apply_gradients(grads)

        # Confidence Score
        alpha_ita = tf.stop_gradient(tf.math.scalar_mul(exp_scale, im_model.vf - im_V))
        weights = tf.clip_by_value(tf.exp(alpha_ita), 0.01, 100)
        weights = weights/tf.reduce_sum(weights)

        # Imitation: MSE
        im_mse_loss = tf.reduce_mean(tf.reduce_mean(tf.square(im_model.pi - im_A), axis=1)* weights)
        im_mse_grads = tf.gradients(im_mse_loss, params)
        if max_grad_norm is not None:
            im_mse_grads, _im_mse_grad_norm = tf.clip_by_global_norm(im_mse_grads, max_grad_norm)
        im_mse_grads = list(zip(im_mse_grads, params))
        _im_mse_train = trainer.apply_gradients(im_mse_grads)

        # Imitation: Negative Loglikelihood
        im_loss = tf.reduce_mean(tf.multiply(im_neglogpac, weights))
        im_grads = tf.gradients(im_loss, params)
        if max_grad_norm is not None:
            im_grads, _im__grad_norm = tf.clip_by_global_norm(im_grads, max_grad_norm)
        im_grads = list(zip(im_grads, params))
        _im_train = trainer.apply_gradients(im_grads)


        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def im_train(lr, obs, actions, values, states=None):
            td_map = {im_model.X:obs, im_A:actions, im_V: values, LR:lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [im_loss, _im_train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        def im_mse_train(lr, obs, actions, values, states=None):
            td_map = {im_model.X: obs, im_A: actions, im_V: values, LR: lr}
            if states is not None:
                td_map[im_model.S] = states
                td_map[im_model.M] = masks
            return sess.run(
                [im_mse_loss, _im_mse_train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        def copy(scope):
            #tvars = tf.trainable_variables()
            #tvars_vals = sess.run(tvars)

            #for var, val in zip(tvars, tvars_vals):
            #    if var.name == 'model1/pi_fc1/w:0' or var.name == 'model0/pi_fc1/w:0':
            #        print(var.name, val)
            e1_params = [t for t in tf.trainable_variables() if t.name.startswith(scope)]
            e1_params = sorted(e1_params, key=lambda v: v.name)
            e2_params = [t for t in tf.trainable_variables() if t.name.startswith('model'+str(policy_scope))]
            e2_params = sorted(e2_params, key=lambda v: v.name)
            update_ops = []
            for e1_v, e2_v in zip(e1_params, e2_params):
                op = e2_v.assign(e1_v)
                update_ops.append(op)

            sess.run(update_ops)
            #print('...........................')

            #tvars = tf.trainable_variables()
            #tvars_vals = sess.run(tvars)

            #for var, val in zip(tvars, tvars_vals):
            #    if var.name == 'model1/pi_fc1/w:0' or var.name == 'model0/pi_fc1/w:0':
            #        print(var.name, val)

        self.train = train
        self.im_train = im_train
        self.im_mse_train = im_mse_train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.im_step = im_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.copy = copy
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam, memory):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.memory = memory

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            self.memory.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

class Learner(object):
    def __init__(self, *, policy, env, nsteps, total_timesteps, ent_coef, lr,
                vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
                log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
                save_interval=0, load_path=None, policy_scope='', value_scope='', scope='', im_batch=512, exp_scale=0.5):

        self.policy = policy
        self.env = env
        self.nsteps = nsteps
        self.total_timesteps = total_timesteps
        self.ent_coef = ent_coef
        self.lr = lr
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.log_interval = log_interval
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.save_interval = save_interval
        self.load_path = load_path
        self.scope = scope
        
        if isinstance(self.lr, float): self.lr = constfn(self.lr)
        else: assert callable(self.lr)
        if isinstance(self.cliprange, float): self.cliprange = constfn(self.cliprange)
        else: assert callable(self.cliprange)
        self.total_timesteps = int(self.total_timesteps)

        self.nenvs = env.num_envs
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.nbatch = self.nenvs * self.nsteps
        self.nbatch_train = self.nbatch // self.nminibatches

        make_model = lambda : Model(policy=policy, ob_space=self.ob_space, ac_space=self.ac_space, nbatch_act=self.nenvs, nbatch_train=self.nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm, policy_scope=policy_scope, value_scope=value_scope, scope=scope, im_batch=im_batch, exp_scale=exp_scale)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        self.model = make_model()
        if load_path is not None:
            model.load(load_path)
        self.memory = Memory(100000, self.ob_space.shape)
        self.runner = Runner(env=env, model=self.model, nsteps=nsteps, gamma=gamma, lam=lam, memory=self.memory)

        self.epinfobuf = deque(maxlen=100)
        self.tfirststart = time.time()

        self.nupdates = self.total_timesteps//self.nbatch

        self.update_index = 1

    #for update in range(1, nupdates+1):
    def update(self):
        if self.update_index > self.nupdates:
            return False
        assert self.nbatch % self.nminibatches == 0
        self.nbatch_train = self.nbatch // self.nminibatches
        tstart = time.time()
        frac = 1.0 - (self.update_index - 1.0) /self.nupdates
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run() #pylint: disable=E0632
        self.epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(self.nbatch)
            for _ in range(self.noptepochs):
                np.random.shuffle(inds)
                for start in range(0, self.nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(self.model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert self.nenvs % self.nminibatches == 0
            envsperbatch = self.nenvs // self.nminibatches
            envinds = np.arange(self.nenvs)
            flatinds = np.arange(self,nenvs * self.nsteps).reshape(self.nenvs, self.nsteps)
            envsperbatch = self.nbatch_train // self.nsteps
            for _ in range(self.noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, self.nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(self.nbatch / (tnow - tstart))
        if self.update_index % self.log_interval == 0 or self.update_index == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", self.update_index*self.nsteps)
            logger.logkv("nupdates", self.update_index)
            logger.logkv("total_timesteps", self.update_index*self.nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in self.epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in self.epinfobuf]))
            logger.logkv('time_elapsed', tnow - self.tfirststart)
            logger.logkv('agent', self.scope)
            for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if self.save_interval and (self.update % self.save_interval == 0 or self.update_index == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%self.update_index)
            print('Saving to', savepath)
            self.model.save(savepath)
        self.update_index += 1
        self.min_reward = safemin([epinfo['r'] for epinfo in self.epinfobuf])
        self.max_reward = safemax([epinfo['r'] for epinfo in self.epinfobuf])
        return True
    #env.close()
    #return model

    def imitate(self, obs, actions, values):
        frac = 1.0 - (self.update_index - 1.0) /self.nupdates
        lrnow = self.lr(frac)
        loss = self.model.im_train(lrnow, obs, actions, values)

    def imitate_mse(self, obs, actions, values):
        frac = 1.0 - (self.update_index - 1.0) /self.nupdates
        lrnow = self.lr(frac)
        loss = self.model.im_mse_train(lrnow, obs, actions, values)

    def sample(self, batch_size):
        obs = self.memory.sample(batch_size)
        actions, values, states, neglogpacs = self.model.im_step(obs)
        return obs, actions, values

    def copy(self, scope):
        self.model.copy(scope)

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def safemin(xs):
    return np.nan if len(xs) == 0 else np.min(xs)

def safemax(xs):
    return np.nan if len(xs) == 0 else np.max(xs)

