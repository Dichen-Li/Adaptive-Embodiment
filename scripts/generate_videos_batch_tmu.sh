for i in {0..330}
do
  /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py --video --headless --num_envs 2 --task Genhumanoid$i
  echo "===============================$i done==============================="
done
