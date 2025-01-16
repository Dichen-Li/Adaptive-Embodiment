import os
import time

# Configuration
task_indices = list(range(168))
tasks_prefix = "Gendog"
tasks_suffix = ""  # Add any suffix if needed
tasks_per_job = 21  # Number of tasks per job
num_parallel_commands = 4  # Number of parallel commands per job
job_name_template = "dichen-job-quadruped-{job_index}-play-collect-data-jan15"
output_folder = "jobs_play_collect_data"  # Folder to store YAML files
submission_script = "submit_jobs_play_collect_data.sh"  # Batch submission script
deletion_script = "delete_jobs_play_collect_data.sh"  # Batch deletion script
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
              memory: "64Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "12"
              memory: "72Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: dichen-fast-vol
              mountPath: /dichen-fast-vol
            - name: bai-fast-vol  # change this based on your own pvc
              mountPath: /bai-fast-vol # change this based on your own pvc
            - name: liudai-fast-vol  # change this based on your own pvc
              mountPath: /liudai-fast-vol # change this based on your own pvc
            - name: tmu-fast-vol  # change this based on your own pvc
              mountPath: /tmu-fast-vol # change this based on your own pvcs
      volumes:
        - name: dshm # shared memory, required for the multi-worker dataloader
          emptyDir:
            medium: Memory
        - name: dichen-fast-vol
          persistentVolumeClaim:
            claimName: dichen-fast-vol
        - name: bai-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: bai-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
        - name: liudai-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: liudai-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
        - name: tmu-fast-vol  # change this based on your own pvc
          persistentVolumeClaim:
            claimName: tmu-fast-vol  # change this based on your own pvc; you can get the pvc name of your created volume from "kubectl get pvc"
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
                      # - NVIDIA-GeForce-RTX-2080-Ti
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
        f"source ~/.bashrc && cd /bai-fast-vol/code/embodiment-scaling-law && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install setuptools wheel && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install build && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install toml && "   # I know its too much but we need these to avoid werid errors.. 
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e exts/berkeley_humanoid && "
        f"/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e rsl_rl && "  # only for sim2real_learning branch
        f"bash scripts/play_collect_data_batch_nautilus.sh --tasks {' '.join(group)})"
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
submission_script = os.path.join(output_folder, submission_script)
with open(submission_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_file in job_files:
        f.write(f"kubectl create -f {job_file}\n")

# Make the submission script executable
os.chmod(submission_script, 0o755)

# Generate deletion script
deletion_script = os.path.join(output_folder, deletion_script)
with open(deletion_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_name in job_names:
        f.write(f"kubectl delete job {job_name}\n")

# Make the deletion script executable
os.chmod(deletion_script, 0o755)

print(f"Generated {len(job_files)} job YAML files, submission script '{submission_script}', and deletion script '{deletion_script}' in '{output_folder}'.")
