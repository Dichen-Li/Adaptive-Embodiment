
This codebase was built upon the Berkeley Humanoid project, so you might see some trace of the legacy code, but our main program logic
was implemented from scratch in the direct environment style (see [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) for an introduction
to Isaac Lab's Direct Env and Manager-Based Env; basically, direct style is closer to Mujoco and the older Isaac Gym). 

## Installation

- Install Isaac Lab, see
  the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Using a python interpreter that has Isaac sLab installed, install the library with
```
cd embodiment-scaling-law
pip install -e exts/berkeley_humanoid/
```

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

## Understanding the codebase
1. `exts/berkeley_humanoid/berkeley_humanoid/assets`: place where the robot USD files are placed, and the entry for importing
robots. Every robot is treated as an articulation, and its associated properties, such as initial state, joint limit, joint type, 
are defined in `ArticulationCfg` in the files there. 
2. `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/locomotion/locomotion_env.py`: the core simulation logic. The observation space
defined in  `_get_observations` and the reward function is `_get_rewards`. 
3. `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/environments`: place where parameters of the environments 
(for different robots), such as scene, ground, velocity command parameters, are defined.


## Single robot training and testing
There are many build-in robots in the codebase, but if you would like to run experiments using the `GenBot1K` dataset, you need to download it from [here](https://drive.google.com/file/d/1T9D7uOnAsab9wM6gAQSQY9PHTk6zu2PQ/view?usp=sharing), unzip it and move it to the asset folder:
```angular2html
unzip gen_embodiments_1124.zip
mv gen_embodiments_1124 exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0
```
Tensorflow logs will go to `logs/rsl_rl/<task_name>/<job_launch_time>`, 
such as `/home/albert/github/isaac_berkeley_humanoid/logs/rsl_rl/GenDog/2024-11-07_21-35-31`

To train just one robot, run 
```angular2htmlpyt
python scripts/rsl_rl/train.py --task GenDog1
```
If you do not wish to have the visualization open, which could slow down training significantly, you should run
```angular2htmlpyt
python scripts/rsl_rl/train.py --task GenDog1 --headless
```

File `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/environments/__init__.py` contains all the task names that we can run. For example, for `GenBot1k`, we can run `Gendog{i}` where `i` could range from `0` to `307`. 

### Test the learned policy
Run
```angular2html
python scripts/rsl_rl/play.py --task GenDog1
```
It will load the best checkpoint in the latest run the evaluation. 

## Training multiple robots in sequence 
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

## Running batch training jobs on nautilus server
To run many jobs on the nautilus server, we need to batch generate job scripts. Follow this steps:
1. Check out `generation/gen_nautilus_jobs.py`. Adjust parameters, particularly `tasks_per_job` and `num_parallel_commands`.
The former means the number of tasks that will be run in one job, and the second one refers to the number of parallel commands
running parallely in one job. For example, if `num_tasks=300` and `tasks_per_job=30`, then we will 
have ten jobs in total. Then, if `num_parallel_commands=4`, there will be `4` training commands running in every job
simultaneously. I found it's usually good to have `num_parallel_commands>2` if there are more than 8 CPU cores and the GPU is 
at least as good as 3090. Just using `num_parallel_commands=1` causes the training to be bottle-necked by the weak CPU cores. 
2. After running the script with
```angular2html
python gen_nautilus_jobs.py
```
you should see a folder `jobs` containing all the job files. You can run `submit_jobs.sh` to submit them all at once.

## Examine batch training results on nautilus server
After running the training scripts for many robots, we need to batch examine the output logs and checkpoints produced by
the program. Some util scripts have been implemented to ease manual checking.
1. Check `.out` logs. The commands in job scripts should have redirected the training printout to `.out` files, and we 
can check these files to see if there are `Error` or `Traceback`. The following script can do this for you (please
adapt the arg values for your own case):
```angular2html
cd scripts
python check_logs_traceback.py --root ../../jobs-logs/ --keyword Gendog --max-index 308
```
2. Check tensorboard logs. The training programs should produce checkpoints (though, please note that even if IsaacLab 
throws errors, training can still resume sometimes; thus, please check `.out` as well). To check if there are checkpoints
produced for every robot, below is an example command: 
```angular2html
cd scripts
python check_tf_checkpoints.py --root ../logs/rsl_rl --keyword Gendog --max-index 308 --min-epoch 3000
```
The script will look for checkpoints saved after the 3000-th epoch for every robot, and report incomplete logs.


## How to add single robot manually
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


## Adding customized robots in batch using generation script
Taking quadruped as an example, but the specific file names could differ for different robots: 
0. Copy file `/home/albert/github/isaac_berkeley_humanoid/scripts/convert_urdf.py` to `${ISAAC_LAB_PATH}/source/standalone/tools/convert_urdf.py`.
Covner URDF to USD using `scripts/urdf_to_usd.sh` or `scripts/urdf_to_usd_batch.sh`. Here are the example commands:
```angular2html
sh urdf_to_usd.sh ~/Downloads/gen_dog_3_variants/gen_dog_1.urdf ~/Downloads/gen_dog_3_variants/usd/gen_dog_1.usd
sh urdf_to_usd_batch.sh ~/Downloads/gen_dog_3_variants ~/Downloads/gen_dog_3_variants_us    # specify folder 
```
1. Place USD files under `exts/berkeley_humanoid/berkeley_humanoid/assets/Robots`. 
For large robot dataset like `GenBot1K`, we could store the folder somewhere else and create a soft link in the directory pointing to the folder, e.g.
```angular2html
ln -s /folder_absolute_path /absolute_path/exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0
```

2. Add PPO configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/agents/gen_quadruped_1k_ppo_cfg.py`, following the pattern in the file. For adding a large number of robots, run `generation/gen_ppo_cfg.py` to generate these lines automatically:
```angular2html
python generation/gen_ppo_cfg.py 
```
but remember to modify the paths in the file. 

3. Add robot configs to `exts/berkeley_humanoid/berkeley_humanoid/assets/gen_quadrupeds.py`. Note that the configs here might be highly relevant for sim-to-real transfer, e.g., actuator parameters. For adding a large number of robots, run `generation/gen_articulation_cfg.py` to generate these lines automatically:
```angular2html
python generation/gen_articulation_cfg.py
```
but also remember to modify the paths in the file.

4. Add training env configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/humanoid/gen_quadrupeds_env.py`. For adding a large number of robots, run `generation/gen_quadruped_env.py` to generate these lines automatically.

5. Register training envs at `exts/berkeley_humanoid/berkeley_humanoid/assets/__init__.py`. For adding a large number of robots, run `generation/gen_init_registry.py` to generate these lines automatically.

## Common errors
1. Initial state values are integers
```angular2html
    self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Long for the source
```
This is pretty likely due to using integers like `0` as the initial state -- please use `0.0` instead.

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
