
This codebase was built upon the Berkeley Humanoid project, so you might see some trace of the legacy code, but our main program logic
was implemented from scratch in the direct environment style (see [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) for an introduction
to Isaac Lab's Direct Env and Manager-Based Env; basically, direct style is closer to Mujoco and the older Isaac Gym). 

## Installation

- Install Isaac Lab, see
  the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Using a python interpreter that has Isaac sLab installed, install the library

```
cd exts/berkeley_humanoid
python -m pip install -e .
```

## Adding customized robots
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
ln -s {folder_path} exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0
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

## Single robot training and testing
There are many build-in robots in the codebase, but if you would like to run experiments using the `GenBot1K` dataset, you need to download it from [here](https://drive.google.com/file/d/1nPq_osKWaZ_P89GdC27DXqrPYIGqKPCh/view?usp=sharing), unzip it and move it to the asset folder:
```angular2html
unzip gen_embodiments_1124.zip
mv gen_embodiments_1124 exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v0
```

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

## How to train multiple robots in seqence 
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

### Common errors
1. Initial state values are integers
```angular2html
    self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Long for the source
```
This is pretty likely due to using integers like `0` as the initial state -- please use `0.0` instead.

Happy training! 
