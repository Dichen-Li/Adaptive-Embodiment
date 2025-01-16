import os
import time

# Configuration
# task_indices = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 169, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 210, 211, 213, 214, 215, 217, 218, 219, 221, 222, 223, 225, 226, 227, 229, 230, 231, 235, 239, 243, 247, 251, 307, 46, 47, 48, 49, 306]  # Specify a list of task indices
# task_indices = list(range(168, 336))
# task_indices = [169, 173, 177, 183, 184, 185, 186, 187, 188, 189, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]
task_indices = [291, 292, 293, 294, 295] # 182 is already done
tasks_prefix = "Gendog"
tasks_suffix = ""  # Add any suffix if needed
tasks_per_job = 11  # Number of tasks per job
num_parallel_commands = 5  # Number of parallel commands per job
job_name_template = "dichen-job-quadruped-{job_index}-jan15"
output_folder = "jobs"  # Folder to store YAML files
submission_script = "jobs/submit_jobs.sh"  # Batch submission script
deletion_script = "jobs/delete_jobs.sh"  # Batch deletion script
sleep_interval = 50  # Time interval (in seconds) between parallel commands to prevent errors

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
              cpu: "8"
              memory: "14Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "12"
              memory: "20Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: dichen-fast-vol
              mountPath: /dichen-fast-vol
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: dichen-fast-vol
          persistentVolumeClaim:
            claimName: dichen-fast-vol
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
                      - NVIDIA-A100-80GB-PCIe-MIG-1g.10gb
                      - NVIDIA-A100-PCIe-40GB
                      - NVIDIA-A100-PCIe-80GB
                      - NVIDIA-A100-SXM4-80GB
                      - NVIDIA-RTX-A6000
                      - NVIDIA-A40
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
        f"(sleep {i * sleep_interval} && "  # Delay the start of each task group by `i * sleep_interval` seconds
        f"source ~/.bashrc && cd /dichen-fast-vol/code/embodiment-scaling-law && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install setuptools wheel && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install build && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install toml && "   # I know its too much but we need these to avoid werid errors.. 
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e exts/berkeley_humanoid && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e rsl_rl && "  # only for sim2real_learning branch
        f"bash scripts/train_batch_nautilus.sh --tasks {' '.join(group)})"
        for i, group in enumerate(task_groups) if group
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
