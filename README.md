## Environment setup
Set up the environment following the README in the original repo `README-OLD.md`. 

## How to add robot
Taking quadruped as an example, but feel free to create new files and adapt accordingly: 
0. Copy file `/home/albert/github/isaac_berkeley_humanoid/scripts/convert_urdf.py` to `${ISAAC_LAB_PATH}/source/standalone/tools/convert_urdf.py`.
Covner URDF to USD using `scripts/urdf_to_usd.sh` or `scripts/urdf_to_usd_batch.sh`. Here are the example commands:
```angular2html
sh urdf_to_usd.sh ~/Downloads/gen_dog_3_variants/gen_dog_1.urdf ~/Downloads/gen_dog_3_variants/usd/gen_dog_1.usd
sh urdf_to_usd_batch.sh ~/Downloads/gen_dog_3_variants ~/Downloads/gen_dog_3_variants_us    # specify folder 
```
1. Place USD files under `exts/berkeley_humanoid/berkeley_humanoid/assets/Robots`
2. Add PPO configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/agents/rsl_rl_ppo_cfg.py`, following the pattern
3. Add robot configs to `exts/berkeley_humanoid/berkeley_humanoid/assets/generated.py`. Note that the configs here might be highly relevant for sim-to-real transfer, e.g., actuator parameters
4. Add training env configs to `exts/hberkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/gen_dog_direct_env.py`
5. Register training envs at `exts/berkeley_humanoid/berkeley_humanoid/assets/__init__.py`

## Single robot training and testing
To train just one robot, run 
```angular2htmlpyt
python scripts/rsl_rl/train.py --task GenDog1
```
If you do not wish to have the visualization open, which could slow down training significantly, add `--headless` to the command. 
### Visualize the learned policy
```angular2html
python scripts/rsl_rl/play.py --task GenDog1
```

## How to batch training (sequential training of multiple robots)
Do the following:
```angular2html
bash scripts/train_batch.sh --tasks GenDog1 GenDog2 GenDog3 GenDog4 GenDog5
```
where `--tasks` is used to specify task IDs. It is personally recommended to direct the program output to an `.out` file
for future reference, e.g., 
```angular2html
bash scripts/train_batch.sh --tasks GenDog1 GenDog2 GenDog3 GenDog4 GenDog5 > train5.out
```
Tensorflow logs will go to `logs/rsl_rl/<task_name>/<job_launch_time>`, 
such as `/home/albert/github/isaac_berkeley_humanoid/logs/rsl_rl/GenDog/2024-11-07_21-35-31`

## Teacher policy supervised distillation
To supervisely distillate a teacher policy, we generate the input & output dataset from the teacher policy. The input data follows the rule of one policy run them all pattern, which includes description vectors. The output follows the normal action pattern.
To generate the dataset, run
```angular2html
python scripts/rsl_rl/play_record.py --task GenDog1
```
The h5py dataset is stored in logs/rsl_rl/GenDog1/'experiments_name'/h5py_record.

And then we supervise on the loss of teacher & student policy output.
To supervise on the loss, run
```angular2html
python scripts/rsl_rl/train_supervised.py --task GenDog1
```
The student policy is stored in logs/rsl_rl/GenDog1/'experiments_name'/h5py_record.

Happy training! 
