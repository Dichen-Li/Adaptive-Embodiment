
## Preliminary
Our codebase uses IsaacLab's `Direct` style, as opposed to the `ManagerBased` style (see [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) for an introduction). In short, direct style is closer to Mujoco and the older Isaac Gym. 


## Installation
- Install Isaac Lab, see
  the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Install the library `berkeley_humanoid` and our customized version of `rsl_rl`
```
cd embodiment-scaling-law
pip install -e exts/berkeley_humanoid/
pip install -e rsl_rl/
```

## Understanding the codebase
1. `exts/berkeley_humanoid/berkeley_humanoid/assets`: place where the robot USD files are placed, and the entry for importing
robots. Every robot is treated as an articulation, and its associated properties, such as initial state, joint limit, joint type, 
are defined in `ArticulationCfg` in the files there. 
2. `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/locomotion/locomotion_env.py`: the core simulation logic. The observation space
defined in  `_get_observations` and the reward function is `_get_rewards`. 
3. `exts/berkeley_humanoid/berkeley_humanoid/tasks/direct/environments`: place where parameters of the environments 
(for different robots), such as scene, ground, velocity command parameters, are defined.
4. `generation`: scripts for generating nautilus training job scripts, articulation configs, PPO configs, in batch, among many others. 


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


## Single robot training and testing
There are many build-in robots in the codebase, but if you would like to run experiments using the `GenBot1K` dataset, you need to download it from [here](https://drive.google.com/file/d/1i5vlWeobLPVPddSCwh6zwlU_NZl4r-qj/view?usp=sharing), unzip it and move it to the asset folder:
```angular2html
unzip file_name.zip
mv file_name exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v2
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

IMPORTANT: Please modify the `log_dir` in `/home/albert/github/embodiment-scaling-law-sim2real/scripts/train_batch_nautilus.sh`, which is
the directory where all program outputs will be redirected to. If you don't change it to your directory, the files will be written to Bo's storage. 

## Examine batch training results on nautilus server
After running the training scripts for many robots, we need to batch examine the output logs and checkpoints produced by
the program. Some util scripts have been implemented to ease manual checking.
1. Check `.out` logs. The commands in job scripts should have redirected the training printout to `.out` files, and we 
can check these files to see if there are `Error` or `Traceback`. The following script can do this for you (please
adapt the arg values for your own case):
```angular2html
python scripts/check_logs_traceback.py --root /bai-fast-vol/code/jobs-logs --keyword Gendog --max-index 308 --num-lines 10
```
It will check all logs beginning with keyword (`Gendog0`-`Gendog308`) and print out the last `10` lines if there is any error.
If you are unsure what each arg means or want to check what the optional args not specified here, use `--help`. This also applies
to the command detailed below. 

2. Check tensorboard logs. The training programs should produce checkpoints (though, please note that even if IsaacLab 
throws errors, training can still resume sometimes; thus, please check `.out` as well). To check if there are checkpoints
produced for every robot, below is an example command: 
```angular2html
python scripts/check_tf_checkpoints.py --keyword Gendog --max-index 308 --min-epoch 3000
```
The script will look for checkpoints saved after the 3000-th epoch for every robot, and report incomplete runs.

3. Analyze reward values. After training is complete for most robots, we have a notebook for analyzing the reward values recorded in the tensorboard logs (e.g, drawing a histogram). However, our IsaacLab docker image does not have jupyter notebook installed. So, to do this, either
   (1) creat a new pod using the docker `albert01102/torch:2.5.1-cuda12.4-cudnn9-runtime`, then run on server
```aiignore
jupyter notebook --allow-root --port 8889
```
and forward the port to local, by running the below on local machine
```aiignore
kubectl port-forward <pod-name> 8889:8889
```

4. Check tensorboard curves. You can launch tensorboard with
```aiignore
tensorboard --logdir logs/rsl_rl/ --port 6007
```
Then forward the port with
```aiignore
kubectl port-forward <pod-name> 6007:6007
```


## How to add single robot manually
Taking quadruped as an example, but feel free to create new files and adapt accordingly: 
0. Copy file `/home/albert/github/isaac_berkeley_humanoid/scripts/convert_urdf.py` to `${ISAAC_LAB_PATH}/source/standalone/tools/convert_urdf.py`.
Covner URDF to USD using `scripts/urdf_to_usd.sh` or `scripts/urdf_to_usd_batch.sh`. Here are the example commands:
```angular2html
sh urdf_to_usd.sh ~/Downloads/gen_dog_3_variants/gen_dog_1.urdf ~/Downloads/gen_dog_3_variants/usd/gen_dog_1.usd
sh urdf_to_usd_batch.sh ~/Downloads/gen_dog_3_variants ~/Downloads/gen_dog_3_variants_us    # specify folder 
```
1. Place USD files under `exts/berkeley_humanoid/berkeley_humanoid/assets/Robots`
2. Add PPO configs to `exts/berkeley_humanoid/berkeley_humanoid/tasks/configs/algorithm/gen_quadruped_1k_ppo_cfg.py`, following the pattern
3. Add robot configs to `exts/berkeley_humanoid/berkeley_humanoid/assets/gen_quadrupeds.py`. Note that the configs here might be highly relevant for sim-to-real transfer, e.g., actuator parameters
4. Add training env configs to `exts/hberkeley_humanoid/berkeley_humanoid/tasks/configs/environment/gen_quadrupeds_cfg.py`
5. Register training envs at `exts/berkeley_humanoid/berkeley_humanoid/assets/__init__.py`


## Adding customized robots in batch using generation script
Taking quadruped as an example, but the specific file names could differ for different robots: 
0. Copy file `/home/albert/github/isaac_berkeley_humanoid/scripts/convert_urdf.py` to `${ISAAC_LAB_PATH}/source/standalone/tools/convert_urdf.py`.
Covner URDF to USD using `scripts/urdf_to_usd.sh` or `scripts/urdf_to_usd_batch.sh`. Here are the example commands:
```angular2html
sh urdf_to_usd.sh ~/Downloads/gen_dog_3_variants/gen_dog_1.urdf ~/Downloads/gen_dog_3_variants/usd/gen_dog_1.usd
sh urdf_to_usd_batch.sh ~/Downloads/gen_dog_3_variants ~/Downloads/gen_dog_3_variants_us    # specify folder 
```

0.1. Update initial height via 
```aiignore
python generation/update_init_height_sapien.py 
```
You will see `train_cfg_v2.json` in all folders. Please install `sapien===2.2.2` for this. It uses SAPIEN to determine the most suitable initial height for each robot. 

0.2. (Optional) Generate description vector via
```aiignore
python generation/get_description_vector_sapien.py 
```
You will see an additional file in all folders, useful for URMA training.

1. Place USD files under `exts/berkeley_humanoid/berkeley_humanoid/assets/Robots`. 
For large robot dataset like `GenBot1K`, we could store the folder somewhere else and create a soft link in the directory pointing to the folder, e.g.
```angular2html
ln -s /folder_absolute_path /absolute_path/exts/berkeley_humanoid/berkeley_humanoid/assets/Robots/GenBot1K-v2
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

NOTE: For newly generated robots, please run `generation/update_init_height_sapien.py` and `generation/get_description_vector_sapien.py` so that the robot folder contains all necessary information for training and distillation. For the latter, please see the section on `Use SAPIEN` for more details.


## Use SAPIEN
We use SAPIEN to compute description vector, initial robot height, etc. To use these scripts, we first install SAPIEN 2.2.2:
```aiignore
pip install sapien==2.2.2
conda install -y networkx">=2.5"
```
Then we can run `generation/get_description_vector_sapien.py` or `generation/update_init_height_sapien.py`. These
scripts don't have any args, but you need to adjust the paths in the code yourself.

P.S.The following error is okay. Just ignore it:
```aiignore
[2024-12-17 20:39:12.722] [SAPIEN] [error] Cylinder collision is not supported. Replaced with a capsule
```


## Policy distillation
This section introduces how to perform policy distillation. 

[//]: # (The goal is to use one policy observation dataset to supervise a URMA model. But before that, we will need to do sanity check on the URMA model. So we should approach the goal within two phases: 1. extract data from one policy observation to supervise the MLP actor-critic model. 2. use one policy observation to supervise the URMA model. The following explanation goes over the phase reversly.)

#### 0. Place checkpoints under the project root, or train robots to obtain checkpointws
```angular2html
unzip logs.zip
```
Checkpoints should be at `logs/rsl_rl/{robot_name}` where `robot_name` could be `GenDog1`, `GenDog2` etc. Alternatively, train any robot that you prefer and checkpoints will appear in the same directory. 

[//]: # (### One policy observation and URM/re RL actor-critic policies for each of the robots, and the student policy is the URMA model. It can be split into 3 steps.)

#### 1. Data Collection
To generate a dataset for a robot, run the following command, which loads the most
recent checkpoint in the directory and save datasets as `.h5` files in 
that folder. 

[//]: # (We first generate a dataset from the teacher policy. The input data is suitable for URMA model, which includes joint description, joint state, foot description, foot state and general state. The output follows the normal action pattern. Replicate this process for each of the robots.)

[//]: # (To generate the dataset, run)
```angular2html
python scripts/rsl_rl/play_collect_data.py --task Gendog100 --steps 2000 --reward_log_file reward_log_file.json --min_pt_epoch 6999 --headless
```
You may interrupt data collection at any time and the files won't corrupt. 

Dataset directory structure: 
Assume that we have the teacher model checkpoint in `logs/rsl_rl/GenDog2/2024-11-11_12-21-42`. The collected dataset is stored as multiple `h5py` files in `logs/rsl_rl/GenDog2/2024-11-11_12-21-42/h5py_record`. We also store the metadata in a yaml file in the same directory. The metadata includes many simulation and robot parameters, such as the number of joints and various indices needed to construct/decompose the URMA observation vector. We also store the stable expert reward (when all envs are done) into same directory if specify --reward_log_file. We can specify --min_pt_epoch to determine the minimum checkpoints needed to start collecting. 

#### 2. Behavior Cloning
To train a student policy, run the following command, which will load the datasets from all robots and perform supervised learning

[//]: # (After generating the dataset, we combine datasets from different robots to one. After that, we feed the input into the student policy network, and supervised on the the loss between the student action and the dataset output.)
[//]: # (To supervisely train the student model from multiple teacher policies, run)

```angular2html
python scripts/rsl_rl/run_distillation.py \
    --tasks Gendog10_gendog__KneeNum_fl0_fr0_rl0_rr0__ScaleJointLimit_fl0_fr0_rl0_rr0_1_0__Geo_lengthen_calf_0_4 \
            Gendog100_gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_1_2 \
            Gendog200_gendog__KneeNum_fl2_fr2_rl2_rr2__ScaleJointLimit_fl1_fr0_rl1_rr0_0_8__Geo_scale_all_0_8 \
            Gendog50_gendog__KneeNum_fl1_fr1_rl1_rr1__ScaleJointLimit_fl1_fr1_rl0_rr0_1_2__Geo_lengthen_thigh_0_4 \
            Genhexapod1_genhexapod__KneeNum_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0__ScaleJointLimit_l1-0_l2-0_l3-0_l4-0_l5-0_l6-0_1_0__Geo_scale_all_1_2 \
    --model urma \
    --exp_name urma_10_100_200_50_haxa1_randomized_additive_bs512_acc2 \
    --batch_size 512 \
    --lr 3e-4 \
    --num_workers 16 \
    --num_epochs 100 \
    --gradient_acc_steps 2
```
where `model` could also be "rsl_rl_actor", "naive_actor". Please beware that the memory usage increases linearly with `num_workers`. For the above comand that uses 16 workers for data loader, expect a RAM usage of around `40GB`. The example command can also be found at `scripts/rsl_rl/run_distillation.sh`. 

The student policy is stored in `log_dir/{experiments_name_with_timestamp}`. Please note that the script uses a cache to dynamically load `.h5` files from disk. If you think data loading is bottlenecking training, please let Bo know to improve the dataset class.

#### 3. Evaluation
Finally, load the policy network and test it in the simulation environment.

To visualize the trained URMA policy, run
```angular2html
python scripts/rsl_rl/eval_student_model_urma.py --task Gendog10 --ckpt_path log_dir/2024-12-15_17-30-35_Gendog10_100_URMA_v0_reference/best_model.pt 
```
If the model is an MLP, run
```angular2html
python scripts/rsl_rl/eval_student_model_mlp.py --task Gendog10
```

The checkpoints will be under `log_dir/{task-name}`. 

If you wish to store videos, which might slow down simulation, add `--video --video_length 200` and the corresponding video will be stored in the directory: `log_dir/{experiment_name}/one_policy_videos/{task}`. 

Reward values will be printed out. You may note down the rewards for full trajectories to compare between teacher and student model. The teacher model can be run with `scripts/rsl_rl/play.py`, and you will see a simlar printout from the program. Note that the rewards here are likely higher than the one logged by tensorboard during training, because now we are in evaluation mode, i.e., using the mean value predicted by the model as the action, while the action was sampled from a gaussian (predicted mean + a std), which encourages exploration but causes worse reward returns. 

## Robot description analysis
### 1. Generate description file
For Gendog, run,
```angular2html
bash scripts/rsl_rl/eval_urma_description_multi.sh   --checkpoint_path "log_dir/urma_3robot/urma_3robot_10epoch_jan29.pt"   --description_log_file "log_dir/urma_3robot/Gendog0_307_averageEnv_policy_description.json" --robot_name "Gendog" --start_index 0 --end_index 307
```
Output stored to  `log_dir/urma_3robot/Gendog0_307_averageEnv_policy_description.json`
You can download it from https://drive.google.com/drive/folders/1ply5_WjY7E79-YPmdtJDdk4ruNuJwvGp?usp=sharing
### 2. Analyze description using t-sne
Check,
```angular2html
analysis/t-sne.ipynb
```

## Adaptive Configure Learning
The goal is to take in inputs, actions and get configure.


You have to download inputs actions pair h5py file either from trained nautilus folder or from local machine. For each robot, only one h5py file is needed.

You can download the configure files still from https://drive.google.com/drive/folders/1ply5_WjY7E79-YPmdtJDdk4ruNuJwvGp?usp=sharing


For Genhexapod1 and Genhexapod2,
run
```angular2html
python scripts/rsl_rl/train_adaptive_configure.py
```


## Common errors
1. Initial state values are integers
```angular2html
    self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Long for the source
```
This is pretty likely due to using integers like `0` as the initial state -- please use `0.0` instead.
