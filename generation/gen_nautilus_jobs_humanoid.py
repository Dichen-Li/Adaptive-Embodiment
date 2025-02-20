import os

# Configuration
# we have 352 humanoids in GenBot1K-v2
task_indices = sorted([19, 124, 180, 336, 348] + [28, 38, 43, 44, 52, 53, 54, 55, 56, 64, 88, 104, 105, 106, 107, 133, 136, 144, 146, 151, 161, 162, 168, 175, 181, 189, 197, 202, 218, 258, 306, 310, 318, 331, 350])
tasks_prefix = "Genhumanoid"
tasks_suffix = ""  # Add any suffix if needed
tasks_per_job = 4  # Number of tasks per job
num_parallel_commands = 4  # Number of parallel commands per job
job_name_template = "tmu-esl-human-v2-4-round3-{job_index}"
output_folder = "jobs"  # Folder to store YAML files
submission_script = "submit_jobs.sh"  # Batch submission script
deletion_script = "delete_jobs.sh"  # Batch deletion script
run_name = 'v2-4-round3' # added by tmu
wandb_login_key = 'USE_YOUR_WANDB_LOGIN_KEY' # added by tmu
wandb_login_key = '44713a60b687b7a3dbe558ae6ef945cbeacb756e'# tmu API key

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Generate job YAML files
yaml_template = """apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: ucsd-haosulab
spec:
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels:
        nautilus.io/rl: "true"
    spec:
      containers:
        - name: gpu-container
          image: albert01102/cuda12.4.1_ubuntu22.04_embodiment:isaac-v1.1-nodisplay
          command:
            - "/bin/bash"
            - "-c"
          args:
            - |
              {parallel_commands}
          resources:
            requests:
              cpu: "12"
              memory: "20Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "12"
              memory: "20Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: cephfs-fast
              mountPath: /cephfs_fast
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: cephfs-fast
          persistentVolumeClaim:
            claimName: tmu-fast-vol
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-GeForce-RTX-4090
                      - NVIDIA-GeForce-RTX-3090
                      - NVIDIA-RTX-A6000
                      - NVIDIA-A10
  backoffLimit: 0
"""

# Create YAML files and collect job names
job_files = []
job_names = []

for i in range(0, len(task_indices), tasks_per_job):
    # Collect tasks for the current job
    tasks = [f"{tasks_prefix}{task_indices[j]}{tasks_suffix}" for j in range(i, min(i + tasks_per_job, len(task_indices)))]

    # Split tasks into parallel groups
    task_groups = [tasks[j::num_parallel_commands] for j in range(num_parallel_commands)]
    parallel_commands = " &\n              ".join(
        f"sleep {group_idx*120+1} && source ~/.bashrc && /workspace/isaaclab/_isaac_sim/python.sh -m pip install wandb && /workspace/isaaclab/_isaac_sim/python.sh -m wandb login {wandb_login_key} && cd /cephfs_fast/embodiment-scaling-law && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install setuptools wheel && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install build && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install toml && "   # I know its too much but we need these to avoid werid errors.. 
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e exts/berkeley_humanoid && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e rsl_rl && "  # only for sim2real_learning branch
        f"bash scripts/train_batch_nautilus_humanoid.sh --tasks {' '.join(group)} --logger wandb --run_name {run_name}"
        for (group_idx, group) in enumerate(task_groups) if group
    )

    parallel_commands += " & wait"  # Ensure all parallel commands finish before the job exits

    job_index = i // tasks_per_job
    job_name = job_name_template.format(job_index=job_index)
    yaml_content = yaml_template.format(job_name=job_name, parallel_commands=parallel_commands)

    job_file = os.path.join(output_folder, f"{job_name}.yaml")
    with open(job_file, "w") as f:
        f.write(yaml_content)
    job_files.append(job_file)
    job_names.append(job_name)

# Generate submission script
with open(submission_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_file in job_files:
        f.write(f"kubectl create -f {job_file}\n")

# Make the submission script executable
os.chmod(submission_script, 0o755)

# Generate deletion script
with open(deletion_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_name in job_names:
        f.write(f"kubectl delete job {job_name}\n")

# Make the deletion script executable
os.chmod(deletion_script, 0o755)

print(f"Generated {len(job_files)} job YAML files in '{output_folder}', submission script '{submission_script}', and deletion script '{deletion_script}'.")
