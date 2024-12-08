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
