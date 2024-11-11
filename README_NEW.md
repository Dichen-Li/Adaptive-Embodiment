## How to add robot
0. Covner URDF to USD using `scripts/urdf_to_usd.sh` or `scripts/urdf_to_usd_batch.sh`
1. Place USD files under `exts/berkeley_humanoid/berkeley_humanoid/assets/Robots`
2. Add PPO configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/agents/rsl_rl_ppo_cfg.py`, following the pattern
3. Add training env configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/gen_dog_direct_env.py`
4. Register training envs at `exts/berkeley_humanoid/berkeley_humanoid/assets/__init__.py`

## How to batch training (sequential training)
Do the following:
```angular2html
bash scripts/train_batch.sh --tasks GenDog GenDog1 GenDog2 GenDog3 GenDog4 GenDog5
```
where `--tasks` is used to specify task IDs. It is personally recommended to direct the program output to an `.out` file
for future reference, e.g., 
```angular2html
bash scripts/train_batch.sh --tasks GenDog GenDog1 GenDog2 GenDog3 GenDog4 GenDog5 > train5.out
```
Tensorflow logs will go to `logs/rsl_rl/<task_name>/<job_launch_time>`, 
such as `/home/albert/github/isaac_berkeley_humanoid/logs/rsl_rl/GenDog/2024-11-07_21-35-31`

Happy training! 
