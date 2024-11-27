import os

# Configuration
num_tasks = 308  # Total number of tasks
tasks_prefix = "Genhumanoid"
tasks_suffix = ""  # Add any suffix if needed
tasks_per_job = 30  # Number of tasks per job
num_parallel_commands = 4  # Number of parallel commands per job
job_name_template = "bai-job-humanoid-{job_index}"
output_folder = "jobs"  # Folder to store YAML files
submission_script = "submit_jobs.sh"  # Batch submission script

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
    spec:
      containers:
        - name: gpu-container
          image: albert01102/cuda12.4.1_ubuntu22.04_embodiment:isaac-v1.1-nodisplay
          command:
            - "/bin/bash"
            - "-c"
          args:
            - |
              source ~/.bashrc && 
              cd /bai-fast-vol/code/embodiment-scaling-law &&
              /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e exts/berkeley_humanoid &&
              {parallel_commands}
          resources:
            requests:
              cpu: "8"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "32Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: bai-fast-vol
              mountPath: /bai-fast-vol
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: bai-fast-vol
          persistentVolumeClaim:
            claimName: bai-fast-vol
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
for i in range(0, num_tasks, tasks_per_job):
    # Collect tasks for the current job
    tasks = [f"{tasks_prefix}{j}{tasks_suffix}" for j in range(i, min(i + tasks_per_job, num_tasks))]

    # Split tasks into parallel groups
    task_groups = [tasks[j::num_parallel_commands] for j in range(num_parallel_commands)]
    parallel_commands = " &\n              ".join(
        f"bash scripts/train_batch_nautilus.sh --tasks {' '.join(group)}" for group in task_groups if group
    )
    parallel_commands += " & wait"  # Ensure all parallel commands complete

    job_name = job_name_template.format(job_index=i // tasks_per_job)
    yaml_content = yaml_template.format(job_name=job_name, parallel_commands=parallel_commands)

    job_file = os.path.join(output_folder, f"{job_name}.yaml")
    with open(job_file, "w") as f:
        f.write(yaml_content)
    job_files.append(job_file)

# Generate submission script
with open(submission_script, "w") as f:
    f.write("#!/bin/bash\n\n")
    for job_file in job_files:
        f.write(f"kubectl create -f {job_file}\n")

# Make the submission script executable
os.chmod(submission_script, 0o755)

print(f"Generated {len(job_files)} job YAML files in '{output_folder}' and submission script '{submission_script}'.")
