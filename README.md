
# Dual Policy Distillation 

## Installation
1. Please follow the [instructions](https://github.com/openai/mujoco-py/blob/master/README.md) to install the mujoco and mujoco-py


2. Make sure that you have **Python 3.5+** and **pip** installed:
```
git clone https://github.com/datamllab/dual-policy-distillation.git
pip install -r requirements.txt
pip install -e .
	
```

## Example
1. DPD\_DDPG
```
python baselines/dpd_ddpg/main.py --env-id HalfCheetah-v2 --num-timesteps 5000000  --nb-epochs 2500 --dis-batch-size 64 --actor-dis-lr 1e-4 --exp-scale 0.75
```

2. DPD\_PPO
```
python baselines/dpd_ppo/main.py --env HalfCheetah-v2  --num-timesteps 20000000 --exp-scale 0.5
```


## Hyperparameters
1. DPD\_DDPG
```
usage: main.py [-h] [--env-id ENV_ID] [--render-eval] [--no-render-eval]
               [--layer-norm] [--no-layer-norm] [--render] [--no-render]
               [--normalize-returns] [--no-normalize-returns]
               [--normalize-observations] [--no-normalize-observations]
               [--seed SEED] [--critic-l2-reg CRITIC_L2_REG]
               [--batch-size BATCH_SIZE] [--dis-batch-size DIS_BATCH_SIZE]
               [--actor-lr ACTOR_LR] [--actor-dis-lr ACTOR_DIS_LR]
               [--critic-lr CRITIC_LR] [--exp-scale EXP_SCALE] [--popart]
               [--no-popart] [--gamma GAMMA] [--reward-scale REWARD_SCALE]
               [--clip-norm CLIP_NORM] [--nb-epochs NB_EPOCHS]
               [--nb-epoch-cycles NB_EPOCH_CYCLES]
               [--nb-train-steps NB_TRAIN_STEPS]
               [--nb-dis-train-steps NB_DIS_TRAIN_STEPS]
               [--nb-eval-steps NB_EVAL_STEPS]
               [--nb-rollout-steps NB_ROLLOUT_STEPS] [--noise-type NOISE_TYPE]
               [--num-timesteps NUM_TIMESTEPS] [--evaluation]
               [--no-evaluation] [--log_dir LOG_DIR]
```

2. DPD\_PPO
```
usage: main.py [-h] [--env-id ENV_ID] [--seed SEED]
               [--num-timesteps NUM_TIMESTEPS] [--play] [--log-dir LOG_DIR]
               [--exp-scale EXP_SCALE]
```

