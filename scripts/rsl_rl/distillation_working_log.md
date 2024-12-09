## Results from Dichen on Dec 8

### Using subset of One Plicy observation for distillation policy
Training:

```
python scripts/rsl_rl/train_one_policy_observation_supervise_actor_critic.py --task GenDog2 --num_epochs 1000 --exp_name one_policy_observation_extract_to_actor_critic_epoch_1000 --headless
```

Testing in simulation:
```
python scripts/rsl_rl/sim_one_policy_observation_supervise_actor_critic.py --task GenDog2 --new_log_dir logs/rsl_rl/GenDog2/2024-11-11_12-21-42/pt_save_actor_critic/2024-12-08_01-35-14_one_policy_observation_extract_to_actor_critic_epoch_1000
```

Test teacher policy:
```
python scripts/rsl_rl/play.py --task GenDog2
```

Policy architecture:
```
/home/research/anaconda3/envs/isaac_lab/lib/python3.10/site-packages/rsl_rl/modules/actor_critic.py
```

### Using simple observation (teacher policy) for student policy
Check out `train_obs_supervise_actor_critic.py` and `play_actor_critic_collect_obs.py`

### Train One Policy for student policy
`play_collect_data.py`, `sim_after_distill_one_policy.py` and `distill_cross_embodiment.py`

## Notes from Bo on Dec 8
0. Cleaned up the code and renamed files for easier understanding
1. Renamed `sim_one_policy_observation_to_actor_critic.py` to `eval_student_model_mlp.py` and slightly adjusted the args and code. 
2. Renamed `distill_cross_embodiment_model.py` to `run_distillation.py`
3. Renamed `train_one_policy_observation_supervise_actor_critic.py` to `run_distillation_mlp_dichen.py`
4. Renamed `sim_after_distill_one_policy` to `eval_student_model_urma.py`
5. 