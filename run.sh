#pip install -e .

#seeds="11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
#envs="Hopper-v2 HalfCheetah-v2 Walker2d-v2 Swimmer-v2 Ant-v2 Pendulum-v0 Humanoid-v2 HumanoidStandup-v2 InvertedDoublePendulum-v2 InvertedPendulum-v2 Reacher-v2"
#envs="Hopper-v2 HalfCheetah-v2 Walker2d-v2 Swimmer-v2 Pendulum-v0 InvertedDoublePendulum-v2 InvertedPendulum-v2"
#envs="HalfCheetah-v2 Walker2d-v2"
envs="Pendulum-v0"
#envs="HalfCheetah-v2 Walker2d-v2 Swimmer-v2 Ant-v2 Pendulum-v0 Humanoid-v2 HumanoidStandup-v2"
seeds="1"
algs="pos_scale_75"

for alg in $algs
do
	for seed in $seeds
	do
		for env in $envs
		do
			tsp python baselines/dil/main.py --env-id $env --num-timesteps 5000000 --seed $seed --log_dir ./log/$alg/${seed}_${env} --nb-epochs 2500 --dis-batch-size 64 --actor-dis-lr 1e-4 --exp-scale 0.75
		done
	done
done
