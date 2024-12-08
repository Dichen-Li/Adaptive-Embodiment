## Using subset of One Plicy observation for distillation policy
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

## Using simple observation (teacher policy) for student policy
Check out `train_obs_supervise_actor_critic.py` and `play_actor_critic_collect_obs.py`

## Train One Policy for student policy
`play_collect_data.py`, `sim_after_distill_one_policy.py` and `distill_cross_embodiment.py`
